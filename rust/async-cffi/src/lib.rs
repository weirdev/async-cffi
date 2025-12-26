use core::panic;
use std::{ffi::c_void, future::Future, pin::Pin, ptr::NonNull, sync::Arc, task::Waker};

use futures::task::{self, ArcWake};

pub type CWaker = extern "C" fn();
// () -> nullable result pointer
pub type CffiPollFuncT = extern "C" fn() -> *const c_void;

#[derive(Debug, Clone)]
pub struct CWakerWrapper {
    pub waker: CWaker,
}

impl ArcWake for CWakerWrapper {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        (arc_self.waker)();
    }
}

pub struct RustWakerWrapper {
    waker: Box<dyn Fn() + Send + Sync>,
}

impl ArcWake for RustWakerWrapper {
    fn wake_by_ref(arc_self: &Arc<Self>) {
        (arc_self.waker)();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafePtr(pub *const c_void);

// SAFETY: Ensure the pointer is safe to share across threads
unsafe impl Send for SafePtr {}
unsafe impl Sync for SafePtr {}

pub struct CffiFuture {
    // Locking just so Rust knows this is thread-safe.
    // Since we should never be polling this from multiple threads at once,
    // just an unsafe cell would be enough.
    // Using a std Mutex so tokio doesn't complain about blocking the thread
    // if we are using this in a tokio context.
    waker: std::sync::Mutex<Option<Waker>>,
    // CffiFuture_ptr (for waker) -> result pointer
    poll_fn: Box<dyn FnMut() -> SafePtr + Send>,
    pub debug: bool,
}

unsafe impl Send for CffiFuture {}
unsafe impl Sync for CffiFuture {}

// C FFI Future that can be polled from Rust, but uses a C callback to poll.
// Usage:
// let fut = CffiFuture::new(poll_fn);
// let result = fut.await; // This will call the C callback to poll the future.
// // From C
// void* poll_fn() {}
// // Call fut.wake() to wake the future from C when it is ready.
impl CffiFuture {
    pub fn new<F>(poll_fn: F) -> Pin<Box<Self>>
    where
        F: FnMut() -> SafePtr + Send + Sync + 'static,
    {
        Box::pin(CffiFuture {
            waker: std::sync::Mutex::new(None),
            poll_fn: Box::new(poll_fn),
            debug: false,
        })
    }

    // Wrap a Rust Future to create a CffiFuture.
    // The output of the Rust Future must be a non-null pointer.
    // Use `from_rust_future_boxed` if the output may be null.
    pub fn from_rust_nonnull_future<F>(fut: F) -> Pin<Box<CffiFuture>>
    where
        F: Future<Output = *const c_void> + Send + 'static,
    {
        let mut box_fut = Box::pin(fut);
        let mut cffi_fut = Box::pin(CffiFuture {
            waker: std::sync::Mutex::new(None),
            poll_fn: Box::new(move || {
                SafePtr(std::ptr::null()) // Placeholder for the poll function
            }),
            debug: false,
        });

        // Poll fn is self referential, so must be constructed after the CffiFuture is created.
        // TODO: Add test that uses this waker
        let cffi_fut_ptr = SafePtr(cffi_fut.as_mut().get_mut() as *mut CffiFuture as *const c_void);
        cffi_fut.poll_fn = Box::new(move || {
            let waker = Arc::new(RustWakerWrapper {
                waker: Box::new(move || {
                    let cffi_fut_ptr = cffi_fut_ptr;
                    let fut = Pin::new(unsafe {
                        (cffi_fut_ptr.0 as *mut CffiFuture)
                            .as_mut()
                            .expect("CffiFuture cannot be null")
                    });
                    fut.as_ref().wake();
                }),
            });
            let waker = task::waker(waker);

            let mut ctx = std::task::Context::from_waker(&waker);
            match box_fut.as_mut().poll(&mut ctx) {
                std::task::Poll::Ready(result) => SafePtr(result),
                std::task::Poll::Pending => {
                    // If the future is pending, we return a null pointer.
                    SafePtr(std::ptr::null())
                }
            }
        });

        cffi_fut
    }

    // Wrap a Rust Future to create a CffiFuture.
    // The rust future can output any type.
    // If the output is guaranteed to be a non-null pointer,
    // `from_rust_nonnull_future` can be used directly.
    pub fn from_rust_future_boxed<T: 'static>(
        fut: impl Future<Output = T> + Send + 'static,
    ) -> Pin<Box<CffiFuture>> {
        CffiFuture::from_rust_nonnull_future(box_future_output(fut))
    }

    pub fn into_raw(self: Pin<Box<Self>>) -> *mut c_void {
        Box::into_raw(Pin::into_inner(self)) as *mut c_void
    }

    pub fn from_raw(ptr: *mut c_void) -> Pin<&'static mut Self> {
        unsafe {
            Pin::new_unchecked(
                (ptr as *mut CffiFuture)
                    .as_mut()
                    .expect("CffiFuture cannot be null"),
            )
        }
    }

    pub fn poll_inner(self: std::pin::Pin<&mut Self>, waker: &Waker) -> SafePtr {
        if self.debug {
            dbg!("Rust: called poll_inner on CffiFuture");
        }
        let cffi_future = self.get_mut();
        let result = (cffi_future.poll_fn)();
        if !result.0.is_null() {
            if cffi_future.debug {
                dbg!("Rust: CffiFuture is ready, returning result");
            }
            return result;
        } else {
            if cffi_future.debug {
                dbg!("Rust: CffiFuture not ready, registering waker");
            }
            // If the result is null, we need to register the waker.
            let mut waker_lock = cffi_future.waker.lock().unwrap();
            match waker_lock.as_mut() {
                Some(existing_waker) => existing_waker.clone_from(waker),
                None => *waker_lock = Some(waker.clone()),
            }
            SafePtr(std::ptr::null()) // Return null if not ready
        }
    }

    pub fn wake(self: std::pin::Pin<&Self>) {
        if let Some(waker) = self.waker.lock().unwrap().take() {
            if self.debug {
                dbg!("Rust calling waker.wake()");
            }
            waker.wake();
            if self.debug {
                dbg!("Rust called waker.wake()");
            }
        } else {
            // Wake called before poll
            panic!("CffiFuture: No waker to wake up");
        }
    }
}

impl Future for CffiFuture {
    type Output = *const c_void;

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.debug {
            dbg!("Rust: Polling CffiFuture via Future trait");
        }
        let waker = cx.waker();
        let result = self.poll_inner(waker);
        if !result.0.is_null() {
            std::task::Poll::Ready(result.0)
        } else {
            std::task::Poll::Pending
        }
    }
}

pub async fn box_future_output<'a, T>(fut: impl Future<Output = T> + Send + 'a) -> *const c_void {
    let result = fut.await;
    Box::into_raw(Box::new(result)) as *const c_void
}

pub fn waker_from_wrapper_ptr(wrapper: *mut c_void) -> Waker {
    let wrapper = unsafe {
        (wrapper as *mut CWakerWrapper)
            .as_mut()
            .expect("Context cannot be null")
    };
    waker_from_wrapper(wrapper.clone())
}

fn waker_from_wrapper(wrapper: CWakerWrapper) -> Waker {
    let arc_wrapper = Arc::new(wrapper);
    task::waker(arc_wrapper)
}

/// `example_dyn_fn_new() -> ptr<Box<dyn Fn()>>`
pub fn example_dyn_fn_new() -> *mut c_void {
    let dyn_fn: Box<dyn Fn()> = Box::new(|| {
        println!("Hello from the dynamic function!");
    });
    let boxed_dyn_fn = Box::new(dyn_fn);
    Box::into_raw(boxed_dyn_fn) as *mut c_void
}

/// `example_pointer_buffer_new() -> CffiPointerBuffer`
pub fn example_pointer_buffer_new() -> CffiPointerBuffer {
    let pointers: Box<[*const c_void]> = Box::new([box_i32(1), box_i32(2), box_i32(3)]);

    CffiPointerBuffer::from_slice(pointers)
}

/// `call_dyn_fn(dyn_fn: ptr<Box<dyn Fn()>>) -> ()`
pub fn call_dyn_fn(dyn_fn: *mut c_void) {
    let dyn_fn = unsafe {
        (dyn_fn as *mut Box<dyn Fn()>)
            .as_ref()
            .expect("Failed to get dyn_fn")
    };
    dyn_fn();
}

/// `blocking_wait(fut: ptr<CffiFuture<T>>) -> ptr<T>`
pub fn blocking_wait(fut: *mut c_void) -> *const c_void {
    let fut = unsafe {
        (fut as *mut CffiFuture)
            .as_mut()
            .expect("CffiFuture cannot be null")
    };
    let pinned = std::pin::Pin::new(fut);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(pinned)
}

/// `waker_wrapper_new(waker: extern fn()) -> ptr<CWakerWrapper>`
pub fn waker_wrapper_new(waker: CWaker) -> *mut c_void {
    let wrapper = Box::new(CWakerWrapper { waker });
    Box::into_raw(wrapper) as *mut c_void
}

/// `new_cffi_future(poll_fn: extern fn() -> opt_ptr<T>, debug: bool) -> ptr<CffiFuture<T>>`
pub fn new_cffi_future(poll_fn: CffiPollFuncT, debug: bool) -> *mut c_void {
    let mut future = CffiFuture::new(move || {
        let result = poll_fn();
        SafePtr(result)
    });
    future.debug = debug;
    future.into_raw()
}

/// `poll_cffi_future(fut: ptr<CffiFuture<T>>, waker: ptr<CWakerWrapper>) -> opt_ptr<T>`
pub fn poll_cffi_future(fut: *mut c_void, waker: *mut c_void) -> *const c_void {
    let fut = unsafe {
        (fut as *mut CffiFuture)
            .as_mut()
            .expect("CffiFuture cannot be null")
    };
    let fut = Pin::new(fut);
    let waker = waker_from_wrapper_ptr(waker);

    if fut.debug {
        dbg!("Rust: Polling CffiFuture from C");
    }

    fut.poll_inner(&waker).0
}

/// `wake_cffi_future(fut: ptr<CffiFuture<T>>) -> ()`
pub fn wake_cffi_future(fut: *mut c_void) {
    let fut = unsafe {
        (fut as *mut CffiFuture)
            .as_mut()
            .expect("CffiFuture cannot be null")
    };
    let pinned = std::pin::Pin::new(fut);
    pinned.as_ref().wake();
}

/// `box_i32(value: i32) -> ptr<i32>`
pub fn box_i32(value: i32) -> *mut c_void {
    let boxed_value = Box::new(value);
    Box::into_raw(boxed_value) as *mut c_void
}

/// `box_u64(value: u64) -> ptr<u64>`
pub fn box_u64(value: u64) -> *mut c_void {
    let boxed_value = Box::new(value);
    Box::into_raw(boxed_value) as *mut c_void
}

/// `box_ptr(value: opt_ptr<T>) -> ptr<opt_ptr<T>>`
pub fn box_ptr(value: *const c_void) -> *mut c_void {
    let boxed_value = Box::new(value);
    Box::into_raw(boxed_value) as *mut c_void
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CffiPointerBuffer {
    pub pointers: *const *const c_void,
    pub length: usize,
}

impl CffiPointerBuffer {
    pub fn as_slice(&self) -> &[*const c_void] {
        unsafe { std::slice::from_raw_parts(self.pointers, self.length) }
    }

    pub fn from_slice(pointers: Box<[*const c_void]>) -> Self {
        let length = pointers.len();
        let buffer = Self {
            pointers: pointers.as_ptr(),
            length,
        };
        let _ = Box::into_raw(pointers);

        buffer
    }

    pub fn new_empty() -> Self {
        Self {
            pointers: NonNull::dangling().as_ptr(),
            length: 0,
        }
    }
}

unsafe impl Send for CffiPointerBuffer {}
unsafe impl Sync for CffiPointerBuffer {}
