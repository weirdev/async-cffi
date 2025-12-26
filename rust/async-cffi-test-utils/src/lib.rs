use std::ffi::c_void;

use async_cffi::{CWaker, CffiPointerBuffer, CffiPollFuncT};

/// `example_dyn_fn_new() -> ptr<Box<dyn Fn()>>`
#[unsafe(no_mangle)]
pub extern "C" fn example_dyn_fn_new() -> *mut c_void {
    async_cffi::example_dyn_fn_new()
}

/// `example_pointer_buffer_new() -> CffiPointerBuffer`
#[unsafe(no_mangle)]
pub extern "C" fn example_pointer_buffer_new() -> CffiPointerBuffer {
    async_cffi::example_pointer_buffer_new()
}

/// `call_dyn_fn(dyn_fn: ptr<Box<dyn Fn()>>) -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn call_dyn_fn(dyn_fn: *mut c_void) {
    async_cffi::call_dyn_fn(dyn_fn);
}

/// `blocking_wait(fut: ptr<CffiFuture<T>>) -> ptr<T>`
#[unsafe(no_mangle)]
pub extern "C" fn blocking_wait(fut: *mut c_void) -> *const c_void {
    async_cffi::blocking_wait(fut)
}

/// `waker_wrapper_new(waker: extern fn()) -> ptr<CWakerWrapper>`
#[unsafe(no_mangle)]
pub extern "C" fn waker_wrapper_new(waker: CWaker) -> *mut c_void {
    async_cffi::waker_wrapper_new(waker)
}

/// `new_cffi_future(poll_fn: extern fn() -> opt_ptr<T>, debug: bool) -> ptr<CffiFuture<T>>`
#[unsafe(no_mangle)]
pub extern "C" fn new_cffi_future(poll_fn: CffiPollFuncT, debug: bool) -> *mut c_void {
    async_cffi::new_cffi_future(poll_fn, debug)
}

/// `poll_cffi_future(fut: ptr<CffiFuture<T>>, waker: ptr<CWakerWrapper>) -> opt_ptr<T>`
#[unsafe(no_mangle)]
pub extern "C" fn poll_cffi_future(fut: *mut c_void, waker: *mut c_void) -> *const c_void {
    async_cffi::poll_cffi_future(fut, waker)
}

/// `wake_cffi_future(fut: ptr<CffiFuture<T>>) -> ()`
#[unsafe(no_mangle)]
pub extern "C" fn wake_cffi_future(fut: *mut c_void) {
    async_cffi::wake_cffi_future(fut)
}

/// `box_i32(value: i32) -> ptr<i32>`
#[unsafe(no_mangle)]
pub extern "C" fn box_i32(value: i32) -> *mut c_void {
    async_cffi::box_i32(value)
}

/// `box_u64(value: u64) -> ptr<u64>`
#[unsafe(no_mangle)]
pub extern "C" fn box_u64(value: u64) -> *mut c_void {
    async_cffi::box_u64(value)
}

/// `box_ptr(value: opt_ptr<T>) -> ptr<opt_ptr<T>>`
#[unsafe(no_mangle)]
pub extern "C" fn box_ptr(value: *const c_void) -> *mut c_void {
    async_cffi::box_ptr(value)
}

#[cfg(test)]
mod tests {
    use std::{pin::Pin, sync::Arc, task::Poll, thread};

    use async_cffi::{CffiFuture, SafePtr, waker_from_wrapper_ptr};
    use futures::task::{self, ArcWake};
    use serial_test::serial;

    use super::*;

    #[test]
    fn test_waker_wrapper_new() {
        static WAKER_CALLED: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
        extern "C" fn dummy_waker() {
            *WAKER_CALLED.lock().unwrap() = true;
        }
        let wrapper_ptr = waker_wrapper_new(dummy_waker);
        assert!(!wrapper_ptr.is_null());
        let waker = waker_from_wrapper_ptr(wrapper_ptr);
        waker.wake();
        assert!(*WAKER_CALLED.lock().unwrap());
    }

    #[test]
    fn test_cffi_future() {
        static POLL_RESULT: std::sync::Mutex<SafePtr> =
            std::sync::Mutex::new(SafePtr(std::ptr::null()));
        static WAKER_MONITOR: std::sync::Mutex<i32> = std::sync::Mutex::new(0);

        extern "C" fn dummy_poll_fn() -> *const c_void {
            println!("C callback called");
            POLL_RESULT.lock().unwrap().0
        }

        let fut = CffiFuture::new(|| SafePtr(dummy_poll_fn()));
        let fut_ptr = fut.into_raw();
        let mut fut = Pin::new(unsafe {
            (fut_ptr as *mut CffiFuture)
                .as_mut()
                .expect("CffiFuture cannot be null")
        });

        struct RustWakerWrapper {
            waker: Box<dyn Fn() + Send + Sync>,
        }
        impl ArcWake for RustWakerWrapper {
            fn wake_by_ref(arc_self: &Arc<Self>) {
                println!("Rust waker called");
                (arc_self.waker)();
            }
        }

        let waker = Arc::new(RustWakerWrapper {
            waker: Box::new(|| {
                *WAKER_MONITOR.lock().unwrap() = 1;
            }),
        });
        let waker = task::waker(waker);
        let mut ctx = std::task::Context::from_waker(&waker);

        // Initial poll should be pending
        if let Poll::Ready(_) = fut.as_mut().poll(&mut ctx) {
            std::panic!("Future should not be ready yet");
        }

        // C callback sets the result
        POLL_RESULT.lock().unwrap().0 = box_i32(42);

        // Simulate the C callback waking the future
        wake_cffi_future(fut_ptr);
        assert_eq!(*WAKER_MONITOR.lock().unwrap(), 1);

        // Now the future should be ready
        if let Poll::Ready(result) = fut.as_mut().poll(&mut ctx) {
            assert!(!result.is_null());
            let result = unsafe {
                let boxed = Box::from_raw(result as *mut i32);
                *boxed
            };
            assert_eq!(result, 42);
        } else {
            std::panic!("Future should be ready now");
        }
    }

    #[tokio::test]
    #[serial]
    async fn test_awaiting_c_cffi_future() {
        static POLL_RESULT: std::sync::Mutex<SafePtr> =
            std::sync::Mutex::new(SafePtr(std::ptr::null()));

        extern "C" fn dummy_poll_fn() -> *const c_void {
            println!("C callback called");
            POLL_RESULT.lock().unwrap().0
        }

        let fut_ptr = new_cffi_future(dummy_poll_fn, false);

        // Simulate async work happening in C
        let fut_ptr_safe = SafePtr(fut_ptr);
        let c_work = thread::spawn(move || {
            // Simulate some work
            thread::sleep(std::time::Duration::from_millis(5000));

            // Work completed, set the result
            *POLL_RESULT.lock().unwrap() = SafePtr(box_i32(42));
            // Wake the future
            let fut_ptr_safe = fut_ptr_safe;
            wake_cffi_future(fut_ptr_safe.0 as *mut c_void);
        });

        // Now we can await the future in Rust
        let cffi_future = unsafe { &mut *(fut_ptr as *mut CffiFuture) };
        let result = cffi_future.await;

        // Ensure the C work thread has completed
        assert!(c_work.is_finished());

        let result = unsafe {
            let boxed_result = Box::from_raw(result as *mut i32);
            *boxed_result
        };
        assert_eq!(result, 42);
    }

    #[test]
    fn test_cffi_pointer_buffer() {
        let initial_data = vec![1, 2, 3, 4, 5];
        let data = initial_data
            .iter()
            .copied()
            .map(|i| box_i32(i) as *const c_void)
            .collect::<Box<_>>();
        let cffi_buffer = CffiPointerBuffer::from_slice(data);

        let data_from_buffer = cffi_buffer.as_slice();
        let data_from_buffer = data_from_buffer
            .iter()
            .map(|&ptr| unsafe {
                let boxed = (ptr as *mut i32)
                    .as_ref()
                    .expect("pointer in buffer cannot be null");
                *boxed
            })
            .collect::<Vec<_>>();

        assert_eq!(initial_data, data_from_buffer);
    }
}
