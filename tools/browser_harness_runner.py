"""Run browser-harness with a narrow Windows delete-pending compatibility guard.

Keep this in Hermes source rather than editing a staged site-packages directory.
Never retry user code, repeat deletion, hide a permanent error, or change auth.
"""
import runpy
import time


def install_cleanup_guard(ipc):
    """Accept cleanup only when the denied Windows endpoint actually disappears."""
    if not ipc.IS_WINDOWS:
        return
    cleanup = ipc.cleanup_endpoint

    def cleanup_endpoint(name):
        try:
            return cleanup(name)
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32):
                raise
            endpoint = ipc.port_path(name)
            # A second unlink could remove a replacement daemon's endpoint.
            # Observe only; persistent denial or a replacement must fail closed.
            for _ in range(5):
                time.sleep(0.1)
                try:
                    endpoint.stat()
                except FileNotFoundError:
                    return
                except PermissionError:
                    continue
            raise

    ipc.cleanup_endpoint = cleanup_endpoint


def main():
    from browser_harness import _ipc

    install_cleanup_guard(_ipc)
    runpy.run_module("browser_harness.run", run_name="__main__")


if __name__ == "__main__":
    main()
