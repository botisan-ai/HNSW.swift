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

    fn load(
        directory: String,
        basename: String,
        distance: DistanceType,
    ) -> Result<Self, HnswError> {
        let dir_path = Path::new(&directory);
        match distance {
            DistanceType::L1 => {
                let io = Box::new(HnswIo::new(dir_path, &basename));
                let io_ptr = Box::into_raw(io);
                let hnsw: Hnsw<'static, f32, DistL1> = unsafe {
                    (*io_ptr)
                        .load_hnsw()
                        .map_err(|e| HnswError::ReloadError(e.to_string()))?
                };
                Ok(HnswIndexInner::L1(HnswInnerL1 {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: NonNull::new(io_ptr),
                }))
            }
            DistanceType::L2 => {
                let io = Box::new(HnswIo::new(dir_path, &basename));
                let io_ptr = Box::into_raw(io);
                let hnsw: Hnsw<'static, f32, DistL2> = unsafe {
                    (*io_ptr)
                        .load_hnsw()
                        .map_err(|e| HnswError::ReloadError(e.to_string()))?
                };
                Ok(HnswIndexInner::L2(HnswInnerL2 {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: NonNull::new(io_ptr),
                }))
            }
            DistanceType::Cosine => {
                let io = Box::new(HnswIo::new(dir_path, &basename));
                let io_ptr = Box::into_raw(io);
                let hnsw: Hnsw<'static, f32, DistCosine> = unsafe {
                    (*io_ptr)
                        .load_hnsw()
                        .map_err(|e| HnswError::ReloadError(e.to_string()))?
                };
                Ok(HnswIndexInner::Cosine(HnswInnerCosine {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: NonNull::new(io_ptr),
                }))
            }
            DistanceType::Dot => {
                let io = Box::new(HnswIo::new(dir_path, &basename));
                let io_ptr = Box::into_raw(io);
                let hnsw: Hnsw<'static, f32, DistDot> = unsafe {
                    (*io_ptr)
                        .load_hnsw()
                        .map_err(|e| HnswError::ReloadError(e.to_string()))?
                };
                Ok(HnswIndexInner::Dot(HnswInnerDot {
                    hnsw: ManuallyDrop::new(hnsw),
                    io_ptr: NonNull::new(io_ptr),
                }))
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
            HnswIndexInner::L2(inner) => inner.hnsw.parallel_insert(&pairs),
            HnswIndexInner::Cosine(inner) => inner.hnsw.parallel_insert(&pairs),
            HnswIndexInner::Dot(inner) => inner.hnsw.parallel_insert(&pairs),
            HnswIndexInner::L1(inner) => inner.hnsw.parallel_insert(&pairs),
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
        let path = Path::new(&directory);
        match &*guard {
            HnswIndexInner::L2(inner) => inner
                .hnsw
                .file_dump(path, &basename)
                .map_err(|e| HnswError::DumpError(e.to_string()))?,
            HnswIndexInner::Cosine(inner) => inner
                .hnsw
                .file_dump(path, &basename)
                .map_err(|e| HnswError::DumpError(e.to_string()))?,
            HnswIndexInner::Dot(inner) => inner
                .hnsw
                .file_dump(path, &basename)
                .map_err(|e| HnswError::DumpError(e.to_string()))?,
            HnswIndexInner::L1(inner) => inner
                .hnsw
                .file_dump(path, &basename)
                .map_err(|e| HnswError::DumpError(e.to_string()))?,
        };
        Ok(())
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
