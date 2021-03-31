import os
import psutil


def check_ext_mem(ext_mem_dir):
    """
    Compute recursively the memory occupation on disk of ::ext_mem_dir::
    directory.

        Args:
            ext_mem_dir (str): path to the directory.
        Returns:
            ext_mem (float): Occupation size in Megabytes
    """

    ext_mem = sum(
        os.path.getsize(
            os.path.join(dirpath, filename)) for
        dirpath, dirnames, filenames in os.walk(ext_mem_dir)
        for filename in filenames
    ) / (1024 * 1024)

    return ext_mem


def check_ram_usage():
    """
    Compute the RAM usage of the current process.

        Returns:
            mem (float): Memory occupation in Megabytes
    """

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)

    return mem
