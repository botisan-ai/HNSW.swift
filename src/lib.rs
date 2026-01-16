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

#[derive(uniffi::Object)]
pub struct HnswIndexL2 {
    inner: Mutex<HnswInnerL2>,
    dimension: u32,
}

#[uniffi::export]
impl HnswIndexL2 {
    #[uniffi::constructor]
    pub fn new(
        max_nb_connection: u32,
        max_elements: u64,
        max_layer: u32,
        ef_construction: u32,
        dimension: u32,
    ) -> Self {
        let hnsw = Hnsw::new(
            max_nb_connection as usize,
            max_elements as usize,
            max_layer as usize,
            ef_construction as usize,
            DistL2 {},
        );
        Self {
            inner: Mutex::new(HnswInnerL2 {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: None,
            }),
            dimension,
        }
    }

    #[uniffi::constructor]
    pub fn load(directory: String, basename: String, dimension: u32) -> Result<Self, HnswError> {
        let dir_path = Path::new(&directory);
        let io = Box::new(HnswIo::new(dir_path, &basename));
        let io_ptr = Box::into_raw(io);

        let hnsw: Hnsw<'static, f32, DistL2> = unsafe {
            (*io_ptr)
                .load_hnsw()
                .map_err(|e| HnswError::ReloadError(e.to_string()))?
        };

        Ok(Self {
            inner: Mutex::new(HnswInnerL2 {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: NonNull::new(io_ptr),
            }),
            dimension,
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
        guard.hnsw.insert((&data, id as usize));
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
        guard.hnsw.parallel_insert(&pairs);
        Ok(())
    }

    #[uniffi::method]
    pub fn search(&self, query: Vec<f32>, k: u32, ef_search: u32) -> Result<Vec<SearchResult>, HnswError> {
        if query.len() != self.dimension as usize {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: query.len() as u32,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let results = guard.hnsw.search(&query, k as usize, ef_search as usize);
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    #[uniffi::method]
    pub fn len(&self) -> Result<u64, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() as u64)
    }

    #[uniffi::method]
    pub fn is_empty(&self) -> Result<bool, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() == 0)
    }

    #[uniffi::method]
    pub fn get_dimension(&self) -> u32 {
        self.dimension
    }

    #[uniffi::method]
    pub fn save(&self, directory: String, basename: String) -> Result<(), HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let path = Path::new(&directory);
        guard
            .hnsw
            .file_dump(path, &basename)
            .map_err(|e| HnswError::DumpError(e.to_string()))?;
        Ok(())
    }

    #[uniffi::method]
    pub fn set_searching_mode(&self, enabled: bool) -> Result<(), HnswError> {
        let mut guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        guard.hnsw.set_searching_mode(enabled);
        Ok(())
    }
}

#[derive(uniffi::Object)]
pub struct HnswIndexCosine {
    inner: Mutex<HnswInnerCosine>,
    dimension: u32,
}

#[uniffi::export]
impl HnswIndexCosine {
    #[uniffi::constructor]
    pub fn new(
        max_nb_connection: u32,
        max_elements: u64,
        max_layer: u32,
        ef_construction: u32,
        dimension: u32,
    ) -> Self {
        let hnsw = Hnsw::new(
            max_nb_connection as usize,
            max_elements as usize,
            max_layer as usize,
            ef_construction as usize,
            DistCosine {},
        );
        Self {
            inner: Mutex::new(HnswInnerCosine {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: None,
            }),
            dimension,
        }
    }

    #[uniffi::constructor]
    pub fn load(directory: String, basename: String, dimension: u32) -> Result<Self, HnswError> {
        let dir_path = Path::new(&directory);
        let io = Box::new(HnswIo::new(dir_path, &basename));
        let io_ptr = Box::into_raw(io);

        let hnsw: Hnsw<'static, f32, DistCosine> = unsafe {
            (*io_ptr)
                .load_hnsw()
                .map_err(|e| HnswError::ReloadError(e.to_string()))?
        };

        Ok(Self {
            inner: Mutex::new(HnswInnerCosine {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: NonNull::new(io_ptr),
            }),
            dimension,
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
        guard.hnsw.insert((&data, id as usize));
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
        guard.hnsw.parallel_insert(&pairs);
        Ok(())
    }

    #[uniffi::method]
    pub fn search(&self, query: Vec<f32>, k: u32, ef_search: u32) -> Result<Vec<SearchResult>, HnswError> {
        if query.len() != self.dimension as usize {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: query.len() as u32,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let results = guard.hnsw.search(&query, k as usize, ef_search as usize);
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    #[uniffi::method]
    pub fn len(&self) -> Result<u64, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() as u64)
    }

    #[uniffi::method]
    pub fn is_empty(&self) -> Result<bool, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() == 0)
    }

    #[uniffi::method]
    pub fn get_dimension(&self) -> u32 {
        self.dimension
    }

    #[uniffi::method]
    pub fn save(&self, directory: String, basename: String) -> Result<(), HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let path = Path::new(&directory);
        guard
            .hnsw
            .file_dump(path, &basename)
            .map_err(|e| HnswError::DumpError(e.to_string()))?;
        Ok(())
    }

    #[uniffi::method]
    pub fn set_searching_mode(&self, enabled: bool) -> Result<(), HnswError> {
        let mut guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        guard.hnsw.set_searching_mode(enabled);
        Ok(())
    }
}

#[derive(uniffi::Object)]
pub struct HnswIndexDot {
    inner: Mutex<HnswInnerDot>,
    dimension: u32,
}

#[uniffi::export]
impl HnswIndexDot {
    #[uniffi::constructor]
    pub fn new(
        max_nb_connection: u32,
        max_elements: u64,
        max_layer: u32,
        ef_construction: u32,
        dimension: u32,
    ) -> Self {
        let hnsw = Hnsw::new(
            max_nb_connection as usize,
            max_elements as usize,
            max_layer as usize,
            ef_construction as usize,
            DistDot {},
        );
        Self {
            inner: Mutex::new(HnswInnerDot {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: None,
            }),
            dimension,
        }
    }

    #[uniffi::constructor]
    pub fn load(directory: String, basename: String, dimension: u32) -> Result<Self, HnswError> {
        let dir_path = Path::new(&directory);
        let io = Box::new(HnswIo::new(dir_path, &basename));
        let io_ptr = Box::into_raw(io);

        let hnsw: Hnsw<'static, f32, DistDot> = unsafe {
            (*io_ptr)
                .load_hnsw()
                .map_err(|e| HnswError::ReloadError(e.to_string()))?
        };

        Ok(Self {
            inner: Mutex::new(HnswInnerDot {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: NonNull::new(io_ptr),
            }),
            dimension,
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
        guard.hnsw.insert((&data, id as usize));
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
        guard.hnsw.parallel_insert(&pairs);
        Ok(())
    }

    #[uniffi::method]
    pub fn search(&self, query: Vec<f32>, k: u32, ef_search: u32) -> Result<Vec<SearchResult>, HnswError> {
        if query.len() != self.dimension as usize {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: query.len() as u32,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let results = guard.hnsw.search(&query, k as usize, ef_search as usize);
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    #[uniffi::method]
    pub fn len(&self) -> Result<u64, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() as u64)
    }

    #[uniffi::method]
    pub fn is_empty(&self) -> Result<bool, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() == 0)
    }

    #[uniffi::method]
    pub fn get_dimension(&self) -> u32 {
        self.dimension
    }

    #[uniffi::method]
    pub fn save(&self, directory: String, basename: String) -> Result<(), HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let path = Path::new(&directory);
        guard
            .hnsw
            .file_dump(path, &basename)
            .map_err(|e| HnswError::DumpError(e.to_string()))?;
        Ok(())
    }

    #[uniffi::method]
    pub fn set_searching_mode(&self, enabled: bool) -> Result<(), HnswError> {
        let mut guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        guard.hnsw.set_searching_mode(enabled);
        Ok(())
    }
}

#[derive(uniffi::Object)]
pub struct HnswIndexL1 {
    inner: Mutex<HnswInnerL1>,
    dimension: u32,
}

#[uniffi::export]
impl HnswIndexL1 {
    #[uniffi::constructor]
    pub fn new(
        max_nb_connection: u32,
        max_elements: u64,
        max_layer: u32,
        ef_construction: u32,
        dimension: u32,
    ) -> Self {
        let hnsw = Hnsw::new(
            max_nb_connection as usize,
            max_elements as usize,
            max_layer as usize,
            ef_construction as usize,
            DistL1 {},
        );
        Self {
            inner: Mutex::new(HnswInnerL1 {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: None,
            }),
            dimension,
        }
    }

    #[uniffi::constructor]
    pub fn load(directory: String, basename: String, dimension: u32) -> Result<Self, HnswError> {
        let dir_path = Path::new(&directory);
        let io = Box::new(HnswIo::new(dir_path, &basename));
        let io_ptr = Box::into_raw(io);

        let hnsw: Hnsw<'static, f32, DistL1> = unsafe {
            (*io_ptr)
                .load_hnsw()
                .map_err(|e| HnswError::ReloadError(e.to_string()))?
        };

        Ok(Self {
            inner: Mutex::new(HnswInnerL1 {
                hnsw: ManuallyDrop::new(hnsw),
                io_ptr: NonNull::new(io_ptr),
            }),
            dimension,
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
        guard.hnsw.insert((&data, id as usize));
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
        guard.hnsw.parallel_insert(&pairs);
        Ok(())
    }

    #[uniffi::method]
    pub fn search(&self, query: Vec<f32>, k: u32, ef_search: u32) -> Result<Vec<SearchResult>, HnswError> {
        if query.len() != self.dimension as usize {
            return Err(HnswError::DimensionMismatch {
                expected: self.dimension,
                got: query.len() as u32,
            });
        }
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let results = guard.hnsw.search(&query, k as usize, ef_search as usize);
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    #[uniffi::method]
    pub fn len(&self) -> Result<u64, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() as u64)
    }

    #[uniffi::method]
    pub fn is_empty(&self) -> Result<bool, HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        Ok(guard.hnsw.get_nb_point() == 0)
    }

    #[uniffi::method]
    pub fn get_dimension(&self) -> u32 {
        self.dimension
    }

    #[uniffi::method]
    pub fn save(&self, directory: String, basename: String) -> Result<(), HnswError> {
        let guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        let path = Path::new(&directory);
        guard
            .hnsw
            .file_dump(path, &basename)
            .map_err(|e| HnswError::DumpError(e.to_string()))?;
        Ok(())
    }

    #[uniffi::method]
    pub fn set_searching_mode(&self, enabled: bool) -> Result<(), HnswError> {
        let mut guard = self.inner.lock().map_err(|_| HnswError::LockError)?;
        guard.hnsw.set_searching_mode(enabled);
        Ok(())
    }
}

uniffi::setup_scaffolding!();
