use std::collections::HashSet;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use hnsw_rs::api::AnnT;
use hnsw_rs::hnsw::{Hnsw, Neighbour as HnswNeighbour};
use hnsw_rs::hnswio::HnswIo;
use hnsw_rs::prelude::*;

#[derive(Debug, thiserror::Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum HnswError {
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Lock acquisition error")]
    LockError,
    #[error("Index is empty")]
    EmptyIndex,
    #[error("Invalid distance type: expected {expected:?}, got {got:?}")]
    DistanceMismatch {
        expected: DistanceType,
        got: DistanceType,
    },
    #[error("Invalid dimension: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },
    #[error("Reload error: {0}")]
    ReloadError(String),
    #[error("Dump error: {0}")]
    DumpError(String),
}

impl From<std::io::Error> for HnswError {
    fn from(e: std::io::Error) -> Self {
        HnswError::IoError(e.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum DistanceType {
    L2,
    Cosine,
    L1,
    Dot,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

#[derive(Debug, Clone, Copy, uniffi::Record)]
pub struct HnswIndexConfig {
    pub max_nb_connection: u32,
    pub max_elements: u64,
    pub max_layer: u32,
    pub ef_construction: u32,
    pub dimension: u32,
    pub distance: DistanceType,
}

impl From<HnswNeighbour> for SearchResult {
    fn from(n: HnswNeighbour) -> Self {
        SearchResult {
            id: n.d_id as u64,
            distance: n.distance,
        }
    }
}

struct HnswInnerL2 {
    hnsw: ManuallyDrop<Hnsw<'static, f32, DistL2>>,
    io_ptr: Option<NonNull<HnswIo>>,
}

impl Drop for HnswInnerL2 {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.hnsw);
            if let Some(ptr) = self.io_ptr.take() {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
    }
}

unsafe impl Send for HnswInnerL2 {}
unsafe impl Sync for HnswInnerL2 {}

struct HnswInnerCosine {
    hnsw: ManuallyDrop<Hnsw<'static, f32, DistCosine>>,
    io_ptr: Option<NonNull<HnswIo>>,
}

impl Drop for HnswInnerCosine {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.hnsw);
            if let Some(ptr) = self.io_ptr.take() {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
    }
}

unsafe impl Send for HnswInnerCosine {}
unsafe impl Sync for HnswInnerCosine {}

struct HnswInnerDot {
    hnsw: ManuallyDrop<Hnsw<'static, f32, DistDot>>,
    io_ptr: Option<NonNull<HnswIo>>,
}

impl Drop for HnswInnerDot {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.hnsw);
            if let Some(ptr) = self.io_ptr.take() {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
    }
}

unsafe impl Send for HnswInnerDot {}
unsafe impl Sync for HnswInnerDot {}

struct HnswInnerL1 {
    hnsw: ManuallyDrop<Hnsw<'static, f32, DistL1>>,
    io_ptr: Option<NonNull<HnswIo>>,
}

impl Drop for HnswInnerL1 {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.hnsw);
            if let Some(ptr) = self.io_ptr.take() {
                drop(Box::from_raw(ptr.as_ptr()));
            }
        }
    }
}

unsafe impl Send for HnswInnerL1 {}
unsafe impl Sync for HnswInnerL1 {}

fn compact_hnsw<D>(
    hnsw: &Hnsw<'static, f32, D>,
    config: HnswIndexConfig,
    deleted_ids: &[u64],
    make_dist: impl FnOnce() -> D,
) -> Result<Hnsw<'static, f32, D>, HnswError>
where
    D: Distance<f32> + Send + Sync,
{
    if hnsw.get_nb_point() == 0 {
        return Ok(Hnsw::new(
            config.max_nb_connection as usize,
            config.max_elements as usize,
            config.max_layer as usize,
            config.ef_construction as usize,
            make_dist(),
        ));
    }

    let mut deleted: HashSet<usize> = HashSet::with_capacity(deleted_ids.len());
    for &id in deleted_ids {
        let id_usize = usize::try_from(id)
            .map_err(|_| HnswError::IoError("Deleted id exceeds usize range".to_string()))?;
        deleted.insert(id_usize);
    }

    let mut seen: HashSet<usize> = HashSet::new();
    let mut kept_count: u64 = 0;
    for point in hnsw.get_point_indexation().into_iter() {
        let id = point.get_origin_id();
        if deleted.contains(&id) || !seen.insert(id) {
            continue;
        }
        kept_count += 1;
    }

    let max_elements = std::cmp::max(config.max_elements, kept_count);
    let new_hnsw = Hnsw::new(
        config.max_nb_connection as usize,
        max_elements as usize,
        config.max_layer as usize,
        config.ef_construction as usize,
        make_dist(),
    );

    let mut seen: HashSet<usize> = HashSet::new();
    for point in hnsw.get_point_indexation().into_iter() {
        let id = point.get_origin_id();
        if deleted.contains(&id) || !seen.insert(id) {
            continue;
        }
        new_hnsw.insert((point.get_v(), id));
    }

    Ok(new_hnsw)
}

enum HnswIndexInner {
    L1(HnswInnerL1),
    L2(HnswInnerL2),
    Cosine(HnswInnerCosine),
    Dot(HnswInnerDot),
}

impl HnswIndexInner {
    fn new(config: HnswIndexConfig) -> Self {
        match config.distance {
            DistanceType::L1 => HnswIndexInner::L1(HnswInnerL1 {
                hnsw: ManuallyDrop::new(Hnsw::new(
                    config.max_nb_connection as usize,
                    config.max_elements as usize,
                    config.max_layer as usize,
                    config.ef_construction as usize,
                    DistL1 {},
                )),
                io_ptr: None,
            }),
            DistanceType::L2 => HnswIndexInner::L2(HnswInnerL2 {
                hnsw: ManuallyDrop::new(Hnsw::new(
                    config.max_nb_connection as usize,
                    config.max_elements as usize,
                    config.max_layer as usize,
                    config.ef_construction as usize,
                    DistL2 {},
                )),
                io_ptr: None,
            }),
            DistanceType::Cosine => HnswIndexInner::Cosine(HnswInnerCosine {
                hnsw: ManuallyDrop::new(Hnsw::new(
                    config.max_nb_connection as usize,
                    config.max_elements as usize,
                    config.max_layer as usize,
                    config.ef_construction as usize,
                    DistCosine {},
                )),
                io_ptr: None,
            }),
            DistanceType::Dot => HnswIndexInner::Dot(HnswInnerDot {
                hnsw: ManuallyDrop::new(Hnsw::new(
                    config.max_nb_connection as usize,
                    config.max_elements as usize,
                    config.max_layer as usize,
                    config.ef_construction as usize,
                    DistDot {},
                )),
                io_ptr: None,
            }),
        }
    }

    /// Loads the published dump (see `DumpFiles`) through `load_hnsw_with_dist`: hnsw_rs's `load_hnsw` marks the graph
    /// as memory-mapped even when it reads the data into memory, and `file_dump` then never overwrites.
    fn load(
        directory: String,
        basename: String,
        distance: DistanceType,
    ) -> Result<Self, HnswError> {
        let directory = Path::new(&directory);
        let current = DumpFiles {
            directory,
            basename: &basename,
        }
        .current();
        Ok(match distance {
            DistanceType::L1 => {
                let (hnsw, io_ptr) = load_graph(directory, &current, DistL1 {})?;
                HnswIndexInner::L1(HnswInnerL1 { hnsw, io_ptr })
            }
            DistanceType::L2 => {
                let (hnsw, io_ptr) = load_graph(directory, &current, DistL2 {})?;
                HnswIndexInner::L2(HnswInnerL2 { hnsw, io_ptr })
            }
            DistanceType::Cosine => {
                let (hnsw, io_ptr) = load_graph(directory, &current, DistCosine {})?;
                HnswIndexInner::Cosine(HnswInnerCosine { hnsw, io_ptr })
            }
            DistanceType::Dot => {
                let (hnsw, io_ptr) = load_graph(directory, &current, DistDot {})?;
                HnswIndexInner::Dot(HnswInnerDot { hnsw, io_ptr })
            }
        })
    }
}

type LoadedGraph<D> = (ManuallyDrop<Hnsw<'static, f32, D>>, Option<NonNull<HnswIo>>);

/// Reads one dump pair. The `HnswIo` the graph borrows is owned by the caller once this succeeds, and freed here when it
/// fails or hnsw_rs panics.
fn load_graph<D: Distance<f32> + Send + Sync>(
    directory: &Path,
    basename: &str,
    distance: D,
) -> Result<LoadedGraph<D>, HnswError> {
    let io_ptr = Box::into_raw(Box::new(HnswIo::new(directory, basename)));
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        (*io_ptr).load_hnsw_with_dist(distance)
    }));
    match loaded {
        Ok(Ok(hnsw)) => Ok((ManuallyDrop::new(hnsw), NonNull::new(io_ptr))),
        failed => {
            drop(unsafe { Box::from_raw(io_ptr) });
            Err(HnswError::ReloadError(match failed {
                Ok(Err(error)) => error.to_string(),
                _ => "hnsw_rs panicked while reading the dump".to_string(),
            }))
        }
    }
}

/// The steps of `DumpFiles::publish` after the pair is written, so tests can abandon it before each one.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PublishStep {
    SyncPair,
    WritePointer,
    RenamePointer,
    RemoveOthers,
}

enum Publication {
    /// Abandoned by a test as if the process were killed: nothing is cleaned up.
    Killed(HnswError),
    Failed(String),
}

impl From<String> for Publication {
    fn from(message: String) -> Self {
        Publication::Failed(message)
    }
}

/// A dump's files in `directory`. A save writes a complete `<basename>.r<n>.hnsw.{graph,data}` pair, flushes it to disk,
/// and only then renames `<basename>.hnsw.current` to name it; that rename is the only point where the published dump
/// changes, so a failed or killed save leaves the previous pair published. A directory without the pointer holds a dump
/// written before revisions, in the `<basename>.hnsw.{graph,data}` pair.
struct DumpFiles<'a> {
    directory: &'a Path,
    basename: &'a str,
}

impl DumpFiles<'_> {
    const SUFFIXES: [&'static str; 2] = [".hnsw.graph", ".hnsw.data"];

    fn file(&self, name: &str) -> std::path::PathBuf {
        self.directory.join(name)
    }

    fn pointer(&self) -> std::path::PathBuf {
        self.file(&format!("{}.hnsw.current", self.basename))
    }

    fn pointer_draft(&self) -> std::path::PathBuf {
        self.file(&format!("{}.hnsw.current.tmp", self.basename))
    }

    fn pair(&self, basename: &str) -> [std::path::PathBuf; 2] {
        Self::SUFFIXES.map(|suffix| self.file(&format!("{basename}{suffix}")))
    }

    fn revision(&self, basename: &str) -> Option<u64> {
        basename
            .strip_prefix(self.basename)?
            .strip_prefix(".r")?
            .parse()
            .ok()
    }

    /// The revision the pointer names, if both of its files exist.
    fn published(&self) -> Option<String> {
        let revision = std::fs::read_to_string(self.pointer())
            .ok()?
            .trim()
            .to_string();
        self.revision(&revision)?;
        self.pair(&revision)
            .iter()
            .all(|file| file.is_file())
            .then_some(revision)
    }

    /// The pair a load reads: the published revision, else the pair written before revisions.
    fn current(&self) -> String {
        self.published()
            .unwrap_or_else(|| self.basename.to_string())
    }

    /// Writes the next revision with `dump`, publishes it, then deletes every other pair of this basename.
    fn publish(
        &self,
        dump: impl FnOnce(&Path, &str) -> Result<String, String>,
    ) -> Result<(), HnswError> {
        self.publish_until(dump, None)
    }

    /// `publish`, abandoned before `stop` as if the process were killed there: nothing from `stop` on runs, cleanup
    /// included.
    fn publish_until(
        &self,
        dump: impl FnOnce(&Path, &str) -> Result<String, String>,
        stop: Option<PublishStep>,
    ) -> Result<(), HnswError> {
        let killed = |step| {
            (stop == Some(step)).then(|| HnswError::DumpError(format!("killed before {step:?}")))
        };
        let next = self
            .published()
            .and_then(|revision| self.revision(&revision))
            .unwrap_or(0)
            + 1;
        let revision = format!("{}.r{next}", self.basename);
        let written = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            dump(self.directory, &revision)
        }));
        let committed = match written {
            Ok(Ok(name)) if name == revision => {
                if let Some(error) = killed(PublishStep::SyncPair) {
                    return Err(error);
                }
                self.commit(&revision, killed)
            }
            Ok(Ok(name)) => Err(format!("the dump was written as {name}, not {revision}").into()),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err("hnsw_rs panicked while writing the dump".to_string().into()),
        };
        match committed {
            Ok(()) => {}
            Err(Publication::Killed(error)) => return Err(error),
            Err(Publication::Failed(message)) => {
                for file in self.pair(&revision) {
                    let _ = std::fs::remove_file(file);
                }
                let _ = std::fs::remove_file(self.pointer_draft());
                return Err(HnswError::DumpError(message));
            }
        }
        if let Some(error) = killed(PublishStep::RemoveOthers) {
            return Err(error);
        }
        let _ = std::fs::File::open(self.directory).and_then(|directory| directory.sync_all());
        self.remove_all_but(&revision);
        Ok(())
    }

    /// Flushes the written pair, then atomically points the pointer at it. Nothing is published unless this succeeds.
    fn commit(
        &self,
        revision: &str,
        killed: impl Fn(PublishStep) -> Option<HnswError>,
    ) -> Result<(), Publication> {
        use std::io::Write;
        let fail = |error: std::io::Error| Publication::Failed(error.to_string());
        for file in self.pair(revision) {
            std::fs::File::open(file)
                .and_then(|file| file.sync_all())
                .map_err(fail)?;
        }
        if let Some(error) = killed(PublishStep::WritePointer) {
            return Err(Publication::Killed(error));
        }
        let mut draft = std::fs::File::create(self.pointer_draft()).map_err(fail)?;
        draft
            .write_all(format!("{revision}\n").as_bytes())
            .map_err(fail)?;
        draft.sync_all().map_err(fail)?;
        if let Some(error) = killed(PublishStep::RenamePointer) {
            return Err(Publication::Killed(error));
        }
        std::fs::rename(self.pointer_draft(), self.pointer()).map_err(fail)
    }

    /// Deletes this basename's other pairs: earlier revisions, the pair written before revisions, and the
    /// `<basename>-<n>` copies that hnsw_rs wrote when it refused to overwrite a loaded graph's dump.
    fn remove_all_but(&self, revision: &str) {
        let Ok(entries) = std::fs::read_dir(self.directory) else {
            return;
        };
        for name in entries.filter_map(|entry| entry.ok()?.file_name().into_string().ok()) {
            let Some(pair) = Self::SUFFIXES
                .iter()
                .find_map(|suffix| name.strip_prefix(self.basename)?.strip_suffix(suffix))
            else {
                continue;
            };
            let stale = pair.is_empty()
                || pair.strip_prefix(".r").is_some_and(|n| {
                    n.parse::<u64>().is_ok() && pair != &revision[self.basename.len()..]
                })
                || pair
                    .strip_prefix('-')
                    .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()));
            if stale {
                let _ = std::fs::remove_file(self.file(&name));
            }
        }
    }
}

#[derive(uniffi::Object)]
pub struct HnswIndex {
    inner: Mutex<HnswIndexInner>,
    dimension: u32,
    distance: DistanceType,
}

#[uniffi::export]
impl HnswIndex {
    #[uniffi::constructor]
    pub fn new(config: HnswIndexConfig) -> Self {
        let dimension = config.dimension;
        let distance = config.distance;
        Self {
            inner: Mutex::new(HnswIndexInner::new(config)),
            dimension,
            distance,
        }
    }

    #[uniffi::constructor]
    pub fn load(
        directory: String,
        basename: String,
        config: HnswIndexConfig,
    ) -> Result<Self, HnswError> {
        let dimension = config.dimension;
        let distance = config.distance;
        let inner = HnswIndexInner::load(directory, basename, distance)?;
        Ok(Self {
            inner: Mutex::new(inner),
            dimension,
            distance,
        })
    }

    #[uniffi::method]
    pub fn insert(&self, data: Vec<f32>, id: u64) -> Result<(), HnswError> {
        if data.len() != self.dimension as usize {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: data.len() as u32,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        match &*guard {
            HnswIndexInner::L2(inner) => inner.hnsw.insert((&data, id as usize)),
            HnswIndexInner::Cosine(inner) => inner.hnsw.insert((&data, id as usize)),
            HnswIndexInner::Dot(inner) => inner.hnsw.insert((&data, id as usize)),
            HnswIndexInner::L1(inner) => inner.hnsw.insert((&data, id as usize)),
        }
        Ok(())
    }

    #[uniffi::method]
    pub fn insert_batch(&self, data: Vec<Vec<f32>>, ids: Vec<u64>) -> Result<(), HnswError> {
        if data.len() != ids.len() {
            return Err(HnswError::IoError(
                "Data and IDs must have the same length".to_string(),
            ));
        }
        for vec in &data {
            if vec.len() != self.dimension as usize {
                return Err(HnswError::DimensionMismatch {
                    expected: self.dimension,
                    got: vec.len() as u32,
                });
            }
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let pairs: Vec<(&Vec<f32>, usize)> =
            data.iter().zip(ids.iter().map(|&id| id as usize)).collect();
        match &*guard {
            HnswIndexInner::L2(inner) => {
                pairs.iter().for_each(|&(v, id)| inner.hnsw.insert((v, id)))
            }
            HnswIndexInner::Cosine(inner) => {
                pairs.iter().for_each(|&(v, id)| inner.hnsw.insert((v, id)))
            }
            HnswIndexInner::Dot(inner) => {
                pairs.iter().for_each(|&(v, id)| inner.hnsw.insert((v, id)))
            }
            HnswIndexInner::L1(inner) => {
                pairs.iter().for_each(|&(v, id)| inner.hnsw.insert((v, id)))
            }
        }
        Ok(())
    }

    #[uniffi::method]
    pub fn search(
        &self,
        query: Vec<f32>,
        k: u32,
        ef_search: u32,
    ) -> Result<Vec<SearchResult>, HnswError> {
        if query.len() != self.dimension as usize {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: query.len() as u32,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let results = match &*guard {
            HnswIndexInner::L2(inner) => inner.hnsw.search(&query, k as usize, ef_search as usize),
            HnswIndexInner::Cosine(inner) => {
                inner.hnsw.search(&query, k as usize, ef_search as usize)
            }
            HnswIndexInner::Dot(inner) => inner.hnsw.search(&query, k as usize, ef_search as usize),
            HnswIndexInner::L1(inner) => inner.hnsw.search(&query, k as usize, ef_search as usize),
        };
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    #[uniffi::method]
    pub fn len(&self) -> Result<u64, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(match &*guard {
            HnswIndexInner::L2(inner) => inner.hnsw.get_nb_point() as u64,
            HnswIndexInner::Cosine(inner) => inner.hnsw.get_nb_point() as u64,
            HnswIndexInner::Dot(inner) => inner.hnsw.get_nb_point() as u64,
            HnswIndexInner::L1(inner) => inner.hnsw.get_nb_point() as u64,
        })
    }

    #[uniffi::method]
    pub fn is_empty(&self) -> Result<bool, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(match &*guard {
            HnswIndexInner::L2(inner) => inner.hnsw.get_nb_point() == 0,
            HnswIndexInner::Cosine(inner) => inner.hnsw.get_nb_point() == 0,
            HnswIndexInner::Dot(inner) => inner.hnsw.get_nb_point() == 0,
            HnswIndexInner::L1(inner) => inner.hnsw.get_nb_point() == 0,
        })
    }

    #[uniffi::method]
    pub fn get_dimension(&self) -> u32 {
        self.dimension
    }

    #[uniffi::method]
    pub fn save(&self, directory: String, basename: String) -> Result<(), HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let files = DumpFiles {
            directory: Path::new(&directory),
            basename: &basename,
        };
        files.publish(|directory, revision| {
            match &*guard {
                HnswIndexInner::L2(inner) => inner.hnsw.file_dump(directory, revision),
                HnswIndexInner::Cosine(inner) => inner.hnsw.file_dump(directory, revision),
                HnswIndexInner::Dot(inner) => inner.hnsw.file_dump(directory, revision),
                HnswIndexInner::L1(inner) => inner.hnsw.file_dump(directory, revision),
            }
            .map_err(|error| error.to_string())
        })
    }

    #[uniffi::method]
    pub fn set_searching_mode(&self, enabled: bool) -> Result<(), HnswError> {
        let mut guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        match &mut *guard {
            HnswIndexInner::L2(inner) => inner.hnsw.set_searching_mode(enabled),
            HnswIndexInner::Cosine(inner) => inner.hnsw.set_searching_mode(enabled),
            HnswIndexInner::Dot(inner) => inner.hnsw.set_searching_mode(enabled),
            HnswIndexInner::L1(inner) => inner.hnsw.set_searching_mode(enabled),
        }
        Ok(())
    }

    #[uniffi::method]
    pub fn compact(
        &self,
        deleted_ids: Vec<u64>,
        config: HnswIndexConfig,
    ) -> Result<Self, HnswError> {
        if config.dimension != self.dimension {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: config.dimension,
            });
        }
        if config.distance != self.distance {
            return Err(HnswError::DistanceMismatch {
                expected: self.distance,
                got: config.distance,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let inner = match &*guard {
            HnswIndexInner::L2(existing) => {
                let hnsw = compact_hnsw(&existing.hnsw, config, &deleted_ids, || DistL2 {})?;
                HnswIndexInner::L2(HnswInnerL2 {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: None,
                })
            }
            HnswIndexInner::Cosine(existing) => {
                let hnsw = compact_hnsw(&existing.hnsw, config, &deleted_ids, || DistCosine {})?;
                HnswIndexInner::Cosine(HnswInnerCosine {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: None,
                })
            }
            HnswIndexInner::Dot(existing) => {
                let hnsw = compact_hnsw(&existing.hnsw, config, &deleted_ids, || DistDot {})?;
                HnswIndexInner::Dot(HnswInnerDot {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: None,
                })
            }
            HnswIndexInner::L1(existing) => {
                let hnsw = compact_hnsw(&existing.hnsw, config, &deleted_ids, || DistL1 {})?;
                HnswIndexInner::L1(HnswInnerL1 {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: None,
                })
            }
        };
        Ok(Self {
            inner: Mutex::new(inner),
            dimension: config.dimension,
            distance: config.distance,
        })
    }
}

uniffi::setup_scaffolding!();

#[cfg(test)]
mod tests {
    use super::*;
    use hnsw_rs::hnsw::PointId;

    fn cosine_config(dimension: u32) -> HnswIndexConfig {
        HnswIndexConfig {
            max_nb_connection: 16,
            max_elements: 200_000,
            max_layer: 16,
            ef_construction: 200,
            dimension,
            distance: DistanceType::Cosine,
        }
    }

    fn vector(seed_text: &str, spread: f32, dimension: usize) -> Vec<f32> {
        let mut seed = seed_text
            .bytes()
            .fold(14_695_981_039_346_656_037u64, |acc, b| {
                (acc ^ b as u64).wrapping_mul(1_099_511_628_211)
            });
        (0..dimension)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                0.01 + ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * spread
            })
            .collect()
    }

    fn found_ids(index: &HnswIndex, query: Vec<f32>, k: u32) -> HashSet<u64> {
        index
            .search(query, k, 150)
            .unwrap()
            .into_iter()
            .map(|r| r.id)
            .collect()
    }

    #[test]
    fn a_second_point_is_never_stranded_from_search() {
        let kept = vector("kept", 0.006, 1536);
        let fresh = vector("fresh", 0.0078, 1536);
        let query = vec![0.01f32; 1536];
        let stranded = (0..2000)
            .filter(|_| {
                let index = HnswIndex::new(cosine_config(1536));
                index.insert(kept.clone(), 0).unwrap();
                index.insert(fresh.clone(), 1).unwrap();
                found_ids(&index, query.clone(), 150) != HashSet::from([0, 1])
            })
            .count();
        assert_eq!(
            stranded, 0,
            "{stranded} of 2000 two-point indexes lost a point"
        );
    }

    fn top_level(index: &HnswIndex) -> u8 {
        match &*index.inner.lock().unwrap() {
            HnswIndexInner::Cosine(inner) => inner.hnsw.get_max_level_observed(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn every_point_of_a_small_library_is_reachable() {
        let query = vec![0.01f32; 64];
        let mut highest_level = 0;
        for size in 2..=64u64 {
            let index = HnswIndex::new(cosine_config(64));
            for id in 0..size {
                index
                    .insert(vector(&format!("memo-{size}-{id}"), 0.008, 64), id)
                    .unwrap();
            }
            let expected: HashSet<u64> = (0..size).collect();
            assert_eq!(
                found_ids(&index, query.clone(), 150),
                expected,
                "a {size}-memo library lost a point"
            );
            highest_level = highest_level.max(top_level(&index));
        }
        assert!(highest_level >= 1, "no library put a point above layer 0");
    }

    type GraphSnapshot = Vec<(PointId, usize, Vec<Vec<(usize, PointId)>>)>;

    fn snapshot<D: Distance<f32> + Send + Sync>(hnsw: &Hnsw<'static, f32, D>) -> GraphSnapshot {
        hnsw.get_point_indexation()
            .into_iter()
            .map(|point| {
                let layers = point
                    .get_neighborhood_id()
                    .into_iter()
                    .map(|layer| layer.into_iter().map(|n| (n.d_id, n.p_id)).collect())
                    .collect();
                (point.get_point_id(), point.get_origin_id(), layers)
            })
            .collect()
    }

    fn graph(index: &HnswIndex) -> GraphSnapshot {
        match &*index.inner.lock().unwrap() {
            HnswIndexInner::L1(inner) => snapshot(&inner.hnsw),
            HnswIndexInner::L2(inner) => snapshot(&inner.hnsw),
            HnswIndexInner::Cosine(inner) => snapshot(&inner.hnsw),
            HnswIndexInner::Dot(inner) => snapshot(&inner.hnsw),
        }
    }

    #[test]
    fn insert_batch_builds_the_graph_of_one_insert_at_a_time() {
        let mut rng = Lcg(20637);
        let points: Vec<Vec<f32>> = (0..512)
            .map(|_| {
                let v: Vec<f32> = (0..32).map(|_| rng.next() + 0.5).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.into_iter().map(|x| x / norm).collect()
            })
            .collect();
        let ids: Vec<u64> = (0..points.len() as u64).collect();
        let workers = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for distance in [
            DistanceType::L1,
            DistanceType::L2,
            DistanceType::Cosine,
            DistanceType::Dot,
        ] {
            let config = HnswIndexConfig {
                distance,
                ..cosine_config(32)
            };
            let batched = HnswIndex::new(config);
            workers.install(|| {
                assert_eq!(rayon::current_num_threads(), 4);
                batched.insert_batch(points.clone(), ids.clone()).unwrap();
            });
            let one_by_one = HnswIndex::new(config);
            for (point, &id) in points.iter().zip(&ids) {
                one_by_one.insert(point.clone(), id).unwrap();
            }
            assert!(
                graph(&batched) == graph(&one_by_one),
                "{distance:?} insert_batch built a different graph than inserting one point at a time"
            );
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        }
    }

    fn reindex_library(seed: u64) {
        let mut rng = Lcg(seed);
        let centres: Vec<Vec<f32>> = (0..12)
            .map(|_| (0..1536).map(|_| rng.next()).collect())
            .collect();
        let memos: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                centres[i % 12]
                    .iter()
                    .map(|c| c + 0.6 * rng.next())
                    .collect()
            })
            .collect();
        let index = HnswIndex::new(cosine_config(1536));
        for (batch, chunk) in memos.chunks(32).enumerate() {
            let ids = (0..chunk.len() as u64)
                .map(|i| batch as u64 * 32 + i)
                .collect();
            index.insert_batch(chunk.to_vec(), ids).unwrap();
        }
        assert_eq!(index.len().unwrap(), 100);
    }

    #[test]
    #[ignore = "stress: HNSW_STRESS_LIBRARIES=3000 cargo test --release -- --ignored"]
    fn insert_batch_never_hangs_across_many_reindexed_libraries() {
        let libraries: u64 = std::env::var("HNSW_STRESS_LIBRARIES")
            .map(|n| n.parse().unwrap())
            .unwrap_or(3000);
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for seed in 1..=libraries {
                reindex_library(seed);
                done.send(seed).unwrap();
            }
        });
        for seed in 1..=libraries {
            match finished.recv_timeout(std::time::Duration::from_secs(60)) {
                Ok(_) => {}
                Err(_) => panic!("insert_batch hung or panicked on library {seed} of {libraries}"),
            }
        }
    }

    fn dump_files(directory: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".hnsw.graph") || name.ends_with(".hnsw.data"))
            .collect();
        names.sort();
        names
    }

    fn unit(mut point: Vec<f32>) -> Vec<f32> {
        let norm = point.iter().map(|x| x * x).sum::<f32>().sqrt();
        point.iter_mut().for_each(|x| *x /= norm);
        point
    }

    #[test]
    fn a_loaded_index_saves_over_its_own_dump() {
        for distance in [
            DistanceType::L1,
            DistanceType::L2,
            DistanceType::Cosine,
            DistanceType::Dot,
        ] {
            let directory = std::env::temp_dir()
                .join(format!("hnsw-resave-{distance:?}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&directory);
            std::fs::create_dir_all(&directory).unwrap();
            let path = directory.to_string_lossy().into_owned();
            let config = HnswIndexConfig {
                distance,
                ..cosine_config(32)
            };

            let first = HnswIndex::new(config);
            for id in 0..40 {
                first
                    .insert(unit(vector(&format!("point {id}"), 2.0, 32)), id)
                    .unwrap();
            }
            first.save(path.clone(), "hnsw".into()).unwrap();

            let loaded = HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap();
            let saved_after_reload = unit(vector("saved after a reload", 2.0, 32));
            loaded.insert(saved_after_reload.clone(), 40).unwrap();
            loaded.save(path.clone(), "hnsw".into()).unwrap();
            assert_eq!(
                dump_files(&directory),
                ["hnsw.r2.hnsw.data", "hnsw.r2.hnsw.graph"],
                "{distance:?}: saving a loaded index must replace its dump, not add a numbered copy"
            );

            let reloaded = HnswIndex::load(path, "hnsw".into(), config).unwrap();
            assert!(
                found_ids(&reloaded, saved_after_reload, 5).contains(&40),
                "{distance:?}: a point saved after a reload must be there after the next reload"
            );
            std::fs::remove_dir_all(&directory).unwrap();
        }
    }

    #[test]
    fn a_loaded_index_keeps_working_after_its_dump_is_replaced() {
        let directory = std::env::temp_dir().join(format!("hnsw-in-memory-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.to_string_lossy().into_owned();
        let config = cosine_config(32);
        let points: Vec<Vec<f32>> = (0..40)
            .map(|id| unit(vector(&format!("point {id}"), 2.0, 32)))
            .collect();
        let first = HnswIndex::new(config);
        for (id, point) in points.iter().enumerate() {
            first.insert(point.clone(), id as u64).unwrap();
        }
        first.save(path.clone(), "hnsw".into()).unwrap();

        let loaded = HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap();
        std::fs::write(directory.join("hnsw.r1.hnsw.data"), b"replaced").unwrap();
        std::fs::remove_file(directory.join("hnsw.r1.hnsw.graph")).unwrap();
        for (id, point) in points.iter().enumerate() {
            assert!(
                found_ids(&loaded, point.clone(), 5).contains(&(id as u64)),
                "point {id} after its dump was replaced"
            );
        }
        loaded
            .insert(
                unit(vector("saved after the dump was replaced", 2.0, 32)),
                40,
            )
            .unwrap();
        loaded.save(path.clone(), "hnsw".into()).unwrap();

        let reloaded = HnswIndex::load(path, "hnsw".into(), config).unwrap();
        assert_eq!(reloaded.len().unwrap(), 41);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("hnsw-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    /// A saved index of 40 points, loaded back, with a 41st point inserted and not yet saved.
    fn loaded_with_unsaved_point(directory: &Path) -> (HnswIndex, HnswIndexConfig, String) {
        let path = directory.to_string_lossy().into_owned();
        let config = cosine_config(32);
        let first = HnswIndex::new(config);
        for id in 0..40 {
            first
                .insert(unit(vector(&format!("point {id}"), 2.0, 32)), id)
                .unwrap();
        }
        first.save(path.clone(), "hnsw".into()).unwrap();
        let loaded = HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap();
        loaded.insert(unit(vector("unsaved", 2.0, 32)), 40).unwrap();
        (loaded, config, path)
    }

    fn data_files(directory: &Path) -> Vec<std::path::PathBuf> {
        std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.to_string_lossy().ends_with(".hnsw.data"))
            .collect()
    }

    #[test]
    fn a_save_that_cannot_write_the_data_file_keeps_the_previous_dump() {
        let directory = scratch("read-only-data");
        let (loaded, config, path) = loaded_with_unsaved_point(&directory);
        for data in data_files(&directory) {
            set_mode(&data, 0o444);
        }
        set_mode(&directory, 0o555);

        let failed = loaded.save(path.clone(), "hnsw".into());
        set_mode(&directory, 0o755);
        for data in data_files(&directory) {
            set_mode(&data, 0o644);
        }
        assert!(
            matches!(failed, Err(HnswError::DumpError(_))),
            "the failed save returns an error: {failed:?}"
        );
        let previous = HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap();
        assert_eq!(
            previous.len().unwrap(),
            40,
            "the previous dump is still loadable"
        );

        loaded.save(path.clone(), "hnsw".into()).unwrap();
        assert_eq!(
            HnswIndex::load(path, "hnsw".into(), config)
                .unwrap()
                .len()
                .unwrap(),
            41,
            "the live index saves once writing works again"
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_save_with_a_read_only_data_file_keeps_the_previous_dump() {
        let directory = scratch("read-only-file");
        let (loaded, config, path) = loaded_with_unsaved_point(&directory);
        for data in data_files(&directory) {
            set_mode(&data, 0o444);
        }

        let result = loaded.save(path.clone(), "hnsw".into());
        for data in data_files(&directory) {
            set_mode(&data, 0o644);
        }
        let reloaded =
            HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap_or_else(|error| {
                panic!("after a save returning {result:?} the dump must load: {error:?}")
            });
        let count = reloaded.len().unwrap();
        assert!(
            (result.is_ok() && count == 41) || (result.is_err() && count == 40),
            "save returned {result:?}, reload has {count} points"
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    /// The child side of `a_save_killed_at_any_moment_leaves_a_loadable_dump`, run as its own process.
    #[test]
    #[ignore = "run by a_save_killed_at_any_moment_leaves_a_loadable_dump"]
    fn save_crash_child() {
        let Ok(path) = std::env::var("HNSW_CRASH_DIR") else {
            return;
        };
        let config = HnswIndexConfig {
            max_elements: 400_000,
            ..cosine_config(256)
        };
        let index = HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap();
        if std::env::var("HNSW_CRASH_ROLE").as_deref() == Ok("verify") {
            println!("HNSW_CRASH_COUNT={}", index.len().unwrap());
            return;
        }
        for id in 100_000u64.. {
            index
                .insert(unit(vector(&format!("saved in a loop {id}"), 2.0, 256)), id)
                .unwrap();
            index.save(path.clone(), "hnsw".into()).unwrap();
        }
    }

    #[test]
    fn a_save_killed_at_any_moment_leaves_a_loadable_dump() {
        let directory = scratch("killed-save");
        let path = directory.to_string_lossy().into_owned();
        let config = HnswIndexConfig {
            max_elements: 400_000,
            ..cosine_config(256)
        };
        let first = HnswIndex::new(config);
        let points: Vec<Vec<f32>> = (0..3000)
            .map(|id| unit(vector(&format!("point {id}"), 2.0, 256)))
            .collect();
        first.insert_batch(points, (0..3000).collect()).unwrap();
        first.save(path.clone(), "hnsw".into()).unwrap();
        let child = |role: &str| {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "tests::save_crash_child",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("HNSW_CRASH_DIR", &path)
                .env("HNSW_CRASH_ROLE", role)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null());
            command
        };
        let mut broken = Vec::new();
        for trial in 0..40u64 {
            let mut saving = child("save").spawn().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(30 + trial * 7));
            saving.kill().unwrap();
            saving.wait().unwrap();
            let verify = child("verify").output().unwrap();
            let loaded = String::from_utf8_lossy(&verify.stdout).contains("HNSW_CRASH_COUNT=");
            if !verify.status.success() || !loaded {
                broken.push(trial);
            }
        }
        assert!(
            broken.is_empty(),
            "the dump could not be loaded after a kill in trials {broken:?}"
        );
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_save_abandoned_before_any_step_leaves_one_whole_dump_loadable() {
        for step in [
            PublishStep::SyncPair,
            PublishStep::WritePointer,
            PublishStep::RenamePointer,
            PublishStep::RemoveOthers,
        ] {
            let directory = scratch("abandoned-save");
            let (loaded, config, path) = loaded_with_unsaved_point(&directory);
            let files = DumpFiles {
                directory: &directory,
                basename: "hnsw",
            };
            let abandoned = files.publish_until(
                |directory, revision| match &*loaded.inner.lock().unwrap() {
                    HnswIndexInner::Cosine(inner) => inner
                        .hnsw
                        .file_dump(directory, revision)
                        .map_err(|e| e.to_string()),
                    _ => unreachable!(),
                },
                Some(step),
            );
            assert!(abandoned.is_err(), "{step:?}");
            let expected = if step == PublishStep::RemoveOthers {
                41
            } else {
                40
            };
            let after_crash =
                HnswIndex::load(path.clone(), "hnsw".into(), config).unwrap_or_else(|error| {
                    panic!("{step:?}: the dump must load after the save was abandoned: {error:?}")
                });
            assert_eq!(
                after_crash.len().unwrap(),
                expected,
                "{step:?}: the previous pair before publication, the new one after"
            );

            loaded.save(path.clone(), "hnsw".into()).unwrap();
            assert_eq!(
                HnswIndex::load(path, "hnsw".into(), config)
                    .unwrap()
                    .len()
                    .unwrap(),
                41,
                "{step:?}"
            );
            assert_eq!(
                dump_files(&directory).len(),
                2,
                "{step:?}: the next save leaves one pair: {:?}",
                dump_files(&directory)
            );
            std::fs::remove_dir_all(&directory).unwrap();
        }
    }
}
