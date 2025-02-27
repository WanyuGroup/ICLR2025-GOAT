import os, logging
from urllib.request import urlopen


def download_data(url, outfile='', binary=False):
    """
    Downloads mol_data from a URL and returns raw mol_data.

    Parameters
    ----------
    url : str
        URL to get the mol_data from
    outfile : str, optional
        Where to save the mol_data.
    binary : bool, optional
        If true, writes mol_data in binary.
    """
    # Try statement to catch downloads.
    try:
        # Download url using urlopen
        with urlopen(url) as f:
            data = f.read()
        logging.info('Data download success!')
        success = True
    except:
        logging.info('Data download failed!')
        success = False

    if binary:
        # If mol_data is binary, use 'wb' if outputting to file
        writeflag = 'wb'
    else:
        # If mol_data is string, convert to string and use 'w' if outputting to file
        writeflag = 'w'
        data = data.decode('utf-8')

    if outfile:
        logging.info('Saving downloaded mol_data to file: {}'.format(outfile))

        with open(outfile, writeflag) as f:
            f.write(data)

    return data, success


# Check if a string can be converted to an int, without throwing an error.
def is_int(str):
    try:
        int(str)
        return True
    except:
        return False

# Cleanup. Use try-except to avoid race condition.
def cleanup_file(file, cleanup=True):
    if cleanup:
        try:
            os.remove(file)
        except OSError:
            pass
