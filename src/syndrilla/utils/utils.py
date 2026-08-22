import os, sys, yaml, json, csv, torch
import numpy as np
import importlib.util

from collections import OrderedDict
from yamlordereddictloader import SafeDumper
from yamlordereddictloader import SafeLoader
from loguru import logger


def parse_device_dtype(cfg):
    """Resolve (device, dtype) from a decoder/interface cfg dict."""
    device_cfg = cfg.get('device', {})
    device_type = device_cfg.get('device_type', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device_type == 'cuda' and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_cfg.get('device_idx', 0)}")
    else:
        device = torch.device('cpu')
    dtype = torch.__dict__[cfg.get('dtype', 'float64')]
    return device, dtype


class bcolors:
    """
    default color palette
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    red = "#EF553B"
    orange = "#E58606"
    yellow = "#FABD2F"
    green = "#9CC424"
    cyan = '#6FD19F'
    blue = "#FABD2F"
    purple = "#AB82FF"
    gray = "#CCCCCC"
    gray2 = "#999999"
    gray3 = "#666666"
    gray4 = "#333333"

    ResetAll = "\033[0m"
    Bold       = "\033[1m"
    Dim        = "\033[2m"
    Underlined = "\033[4m"
    Blink      = "\033[5m"
    Reverse    = "\033[7m"
    Hidden     = "\033[8m"

    ResetBold       = "\033[21m"
    ResetDim        = "\033[22m"
    ResetUnderlined = "\033[24m"
    ResetBlink      = "\033[25m"
    ResetReverse    = "\033[27m"
    ResetHidden     = "\033[28m"

    Default      = "\033[39m"
    Black        = "\033[30m"
    Red          = "\033[31m"
    Green        = "\033[32m"
    Yellow       = "\033[33m"
    Blue         = "\033[34m"
    Magenta      = "\033[35m"
    Cyan         = "\033[36m"
    LightGray    = "\033[37m"
    DarkGray     = "\033[90m"
    LightRed     = "\033[91m"
    LightGreen   = "\033[92m"
    LightYellow  = "\033[93m"
    LightBlue    = "\033[94m"
    LightMagenta = "\033[95m"
    LightCyan    = "\033[96m"
    White        = "\033[97m"

    BackgroundDefault      = "\033[49m"
    BackgroundBlack        = "\033[40m"
    BackgroundRed          = "\033[41m"
    BackgroundGreen        = "\033[42m"
    BackgroundYellow       = "\033[43m"
    BackgroundBlue         = "\033[44m"
    BackgroundMagenta      = "\033[45m"
    BackgroundCyan         = "\033[46m"
    BackgroundLightGray    = "\033[47m"
    BackgroundDarkGray     = "\033[100m"
    BackgroundLightRed     = "\033[101m"
    BackgroundLightGreen   = "\033[102m"
    BackgroundLightYellow  = "\033[103m"
    BackgroundLightBlue    = "\033[104m"
    BackgroundLightMagenta = "\033[105m"
    BackgroundLightCyan    = "\033[106m"
    BackgroundWhite        = "\033[107m"


def strip_list(input_list: list) -> list:
    """
    strip leading and trailing spaces for each list item
    """
    l = []

    for e in input_list:
        e = e.strip()
        if e != '' and e != ' ':
            l.append(e)

    return l


def check_type(input, type):
    """
    check whether input is the required type
    """
    assert isinstance(input, type), logger.error('Invalid input type')
    

def check_file_list(file_list: list):
    """
    check whether all files in the list exist
    """
    for file in file_list:
        assert os.path.exists(file), logger.error('No file: ' + file)


def clean_file_list(file_list: list):
    """
    delete files in the list, if they exist
    """
    for file in file_list:
        if os.path.exists(file):
            logger.warning('Delete file: ' + file)
            os.remove(file)


def create_dir(directory):
    """
    Checks the existence of a directory, if does not exist, create a new one
    :param directory: path to directory under concern
    :return: None
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.success('Create directory: ' + directory)
    except OSError:
        logger.error('Create directory: ' +  directory)
        sys.exit()
    

def create_subdir(path: str, subdir_list: list):
    for subdir in subdir_list:
        subdir_path = os.path.join(path, subdir.strip('/'))
        if not os.path.exists(subdir_path):
            create_dir(subdir_path)

    
def read_yaml(file):
    return yaml.load(open(file), Loader=SafeLoader)


def write_yaml(file, content):
    """
    if file exists at filepath, overwite the file, if not, create a new file
    :param filepath: string that specifies the destination file path
    :param content: yaml string that needs to be written to the destination file
    :return: None
    """
    if os.path.exists(file):
        os.remove(file)
    create_dir(os.path.dirname(file))
    out_file = open(file, 'a')
    out_file.write(yaml.dump( content, default_flow_style= False, Dumper=SafeDumper))


def check_repeated_key(full_dict: OrderedDict, key:str, val: OrderedDict):
    key_index = list(full_dict.keys()).index(key)
    key_list = list(full_dict.keys())[0 : key_index]
    for key in key_list:
        if full_dict[key] == val:
            return True, key
    return False, None
    

# The following interpolate_oneD_linear and interpolate_oneD_quadratic are adapted from accelergy
# ===============================================================
# useful helper functions that are commonly used in estimators
# ===============================================================
def interpolate_oneD_linear(desired_x, known):
    """
    utility function that performs 1D linear interpolation with a known energy value
    :param desired_x: integer value of the desired attribute/argument
    :param known: list of dictionary [{x: <value>, y: <energy>}]
    :return energy value with desired attribute/argument
    """
    # assume E = ax + c where x is a hardware attribute
    ordered_list = []
    if known[1]['x'] < known[0]['x']:
        ordered_list.append(known[1])
        ordered_list.append(known[0])
    else:
        ordered_list = known

    slope = (known[1]['y'] - known[0]['y']) / (known[1]['x'] - known[0]['x'])
    desired_energy = slope * (desired_x - ordered_list[0]['x']) + ordered_list[0]['y']
    return desired_energy


def interpolate_oneD_quadratic(desired_x, known):
    """
    utility function that performs 1D linear interpolation with a known energy value
    :param desired_x: integer value of the desired attribute/argument
    :param known: list of dictionary [{x: <value>, y: <energy>}]
    :return energy value with desired attribute/argument
    """
    # assume E = ax^2 + c where x is a hardware attribute
    ordered_list = []
    if known[1]['x'] < known[0]['x']:
        ordered_list.append(known[1])
        ordered_list.append(known[0])
    else:
        ordered_list = known

    slope = (known[1]['y'] - known[0]['y']) / (known[1]['x']**2 - known[0]['x']**2)
    desired_energy = slope * (desired_x**2 - ordered_list[0]['x']**2) + ordered_list[0]['y']
    return desired_energy


def get_input_tuple(input, size=2):
    if isinstance(input, tuple):
        assert len(input) == size, logger.error('Invalid input size: ' + str(len(input)) + '!=' + str(size))
        return input
    else:
        output = (input, ) * size
        return output


def get_path(path):
    path = os.path.abspath(path)
    path = os.path.realpath(path)
    assert os.path.exists(path), logger.error('Invalid path: ' + path)
    return path


def save_metrics_to_csv(csv_path, row_dict, fieldnames):
    """
    Save one row of metrics to a CSV file. Creates a new file with a header
    if one doesn't already exist, otherwise appends to the existing file.
    """
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def uniquify_list(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def get_dict(input_dict: OrderedDict):
    return json.loads(json.dumps(input_dict))


def check_dict_in_list(input_dict, input_list):
    return get_dict(input_dict) in input_list


def check_dict_equal(input_dict0, input_dict1):
    return get_dict(input_dict0) == get_dict(input_dict1)


def get_prod(input_array):
    return np.prod(np.array(input_array))


def check_yaml_header(input_dict: OrderedDict, header: str, yaml_path: str):
    assert header in input_dict.keys(), logger.error(f'Missing header <{header}> in .{header}.yaml at <{yaml_path}>.')


def check_yaml_cfg(input_dict: OrderedDict, key: str, yaml_path: str):
    assert key in input_dict.keys(), logger.error(f'Missing key <{key}> in the configuration at <{yaml_path}>.')


def call_func_from_yaml(yaml_path: str=None, header: str=None, func_name: str=None, py_path: str=None, **kwargs):
    full_path = get_path(yaml_path)
    load_cfg = read_yaml(full_path)

    # check yaml header
    check_yaml_header(load_cfg, header, full_path)

    # get config
    load_cfg = load_cfg[header]

    # func_name has to be specified
    check_yaml_cfg(load_cfg, func_name, full_path)
    func = load_cfg[func_name].lower()

    # find proper func_name to create the header
    dst_file = os.path.join(py_path, func, func + '.py')
    spec = importlib.util.spec_from_file_location(f'create_{header}_with_{func}', dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    return module_py.create(load_cfg, **kwargs)


def call_func_from_cfg(cfg: dict, header: str, func_name: str, py_path: str, **kwargs):
    check_yaml_cfg(cfg, func_name, "<in-memory>")
    func = cfg[func_name].lower()

    dst_file = os.path.join(py_path, func, func + '.py')
    spec = importlib.util.spec_from_file_location(f'create_{header}_with_{func}', dst_file)
    module_py = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module_py
    spec.loader.exec_module(module_py)

    return module_py.create(cfg, **kwargs)


class dataset():
    # create a pytorch dataset
    def __init__(self, inputs, llrs, labels):
        self.inputs = inputs
        self.llrs = llrs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        llr = self.llrs[index]
        label = self.labels[index]
        return input, llr, label


# from https://github.com/quantumgizmos/bp_osd/blob/a179e6e86237f4b9cc2c952103fce919da2777c8/src/bposd/css.py
def compute_lz(hx,hz):
    #lz logical operators
    #lz\in ker{hx} AND \notin Im(Hz.T)

    ker_hx=nullspace(hx) #compute the kernel basis of hx
    im_hzT=row_basis(hz) #compute the image basis of hz.T

    #in the below we row reduce to find vectors in kx that are not in the image of hz.T.
    log_stack=np.vstack([im_hzT,ker_hx])
    pivots=row_echelon(log_stack.T)[3]
    log_op_indices=[i for i in range(im_hzT.shape[0],log_stack.shape[0]) if i in pivots]
    log_ops=log_stack[log_op_indices]
    return log_ops


# The following functions are from https://github.com/quantumgizmos/ldpc/blob/main/src/ldpc/mod2.py
# row_echelon
# nullspace
# row_basis
from scipy import sparse
def row_echelon(matrix, full=False):
    
    """
    Converts a binary matrix to row echelon form via Gaussian Elimination

    Parameters
    ----------
    matrix : numpy.ndarry or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
    full: bool, optional
        If set to `True', Gaussian elimination is only performed on the rows below
        the pivot. If set to `False' Gaussian eliminatin is performed on rows above
        and below the pivot. 
    
    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination

    Examples
    --------
    >>> H=np.array([[1, 1, 1],[1, 1, 1],[0, 1, 0]])
    >>> re_matrix=row_echelon(H)[0]
    >>> print(re_matrix)
    [[1 1 1]
     [0 1 0]
     [0 0 0]]

    >>> re_matrix=row_echelon(H,full=True)[0]
    >>> print(re_matrix)
    [[1 0 1]
     [0 1 0]
     [0 0 0]]

    """

    num_rows, num_cols = np.shape(matrix)

    # Take copy of matrix if numpy (why?) and initialise transform matrix to identity
    if isinstance(matrix, np.ndarray):
        the_matrix = np.copy(matrix)
        transform_matrix = np.identity(num_rows).astype(int)
    elif isinstance(matrix, sparse.csr.csr_matrix):
        the_matrix = matrix
        transform_matrix = sparse.eye(num_rows, dtype='int', format='csr')
    else:
        raise ValueError('Unrecognised matrix type')

    pivot_row = 0
    pivot_cols = []

    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(num_cols):

        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:

            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])

            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:

                # Swap rows
                the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]

                # Transformation matrix update to reflect this row swap
                transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:

            if not full:  
                elimination_range = [k for k in range(pivot_row + 1, num_rows)]
            else:
                elimination_range = [k for k in range(num_rows) if k != pivot_row]

            # Let's zero those values below the pivot by adding our current row to their row
            for j in elimination_range:

                if the_matrix[j, col] != 0 and pivot_row != j:    ### Do we need second condition?

                    the_matrix[j] = (the_matrix[j] + the_matrix[pivot_row]) % 2

                    # Update transformation matrix to reflect this op
                    transform_matrix[j] = (transform_matrix[j] + transform_matrix[pivot_row]) % 2

            pivot_row += 1
            pivot_cols.append(col)

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    # The rank is equal to the maximum pivot index
    matrix_rank = pivot_row
    row_esch_matrix = the_matrix

    return [row_esch_matrix, matrix_rank, transform_matrix, pivot_cols]


def nullspace(matrix):
    """
    Computes the nullspace of the matrix M. Also sometimes referred to as the kernel.

    All vectors x in the nullspace of M satisfy the following condition::

        Mx=0 \\forall x \\in nullspace(M)
   
    Notes
    -----
    Why does this work?

    The transformation matrix, P, transforms the matrix M into row echelon form, ReM::

        P@M=ReM=[A,0]^T,
    

    where the width of A is equal to the rank. This means the bottom n-k rows of P
    must produce a zero vector when applied to M. For a more formal definition see
    the Rank-Nullity theorem.

    Parameters
    ----------
    matrix: numpy.ndarray
        A binary matrix in numpy.ndarray format
    
    Returns
    -------
    numpy.ndarray
        A binary matrix where each row is a nullspace vector of the inputted binary
        matrix
    
    Examples
    --------
    >>> H=np.array([[0, 0, 0, 1, 1, 1, 1],[0, 1, 1, 0, 0, 1, 1],[1, 0, 1, 0, 1, 0, 1]])
    >>> print(nullspace(H))
    [[1 1 1 0 0 0 0]
     [0 1 1 1 1 0 0]
     [0 1 0 1 0 1 0]
     [0 0 1 1 0 0 1]]
    """

    transpose = matrix.T
    m, n = transpose.shape
    _, matrix_rank, transform, _ = row_echelon(transpose)
    nspace = transform[matrix_rank:m]
    return nspace


def row_basis(matrix):
    """
    Outputs a basis for the rows of the matrix.


    Parameters
    ----------
    matrix: numpy.ndarray
        The input matrix

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray matrix where each row is a basis element.
    
    Examples
    --------

    >>> H=np.array([[1,1,0],[0,1,1],[1,0,1]])
    >>> rb=row_basis(H)
    >>> print(rb)
    [[1 1 0]
     [0 1 1]]
    """
    return matrix[row_echelon(matrix.T)[3]]

# The following functions are from https://github.com/diwu1990/RAVEN/blob/10d126930ed31056e55803da4f8d606cde2b56d2/pe/appr_utils.py#L31
class RoundingNoGrad(torch.autograd.Function):
    """
    RoundingNoGrad is a rounding operation which bypasses the input gradient to output directly.
    Original round()/floor()/ceil() opertions have a gradient of 0 everywhere, which is not useful 
    when doing approximate computing.
    This is something like the straight-through estimator (STE) for quantization-aware training.
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, mode='round'):
        if mode == 'round':
            return input.round()
        elif mode == 'floor':
            return input.floor()
        elif mode == 'ceil':
            return input.ceil()
        else:
            raise ValueError('Input rounding is not supported.')
    
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None
    
    
def fp2fxp(input, intwidth=7, fracwidth=8, rounding='floor'):
    """
    Trunc is an operation to convert data to format (1, intwidth, fracwidth).
    """
    scale = 2**fracwidth
    max_val = (2**(intwidth + fracwidth) - 1)
    min_val = 0 - (2**(intwidth + fracwidth))
    return RoundingNoGrad.apply(input.mul(scale), rounding).clamp(min_val, max_val).div(scale)


def should_flush_extra_queue(n_extra, num_err, target_error, batch_size, extra_density):
    """predict_pct schedule for the rebatch_speedup deferred ('extra') queue.

    The cap defers hard (slow-to-converge) samples into an extra queue; this decides
    WHEN to re-decode them (NOT the cap percentile, which decides WHICH samples get
    deferred). Flush once the queue is predicted to hold the remaining errors still
    needed, using g = `extra_density`, the hard-queue error density (errors per drained
    sample) measured once from the FIRST extra batch and then held fixed:
      - extra_density is None (no first extra yet):    queue >= remaining
      - extra_density == 0   (first batch found none): queue >= batch_size
      - else:                                          queue >= min(remaining / g, batch_size)
    The phase-2 threshold is capped at batch_size (never wait for more than one batch).
    Plus an endgame flush once the error budget is passed.

    Returns (flush, flushing): whether to run an extra batch now, and whether this is the
    endgame drain (used only for logging).
    """
    flushing = num_err >= batch_size                      # endgame: error budget passed
    if n_extra <= 0:
        return False, flushing
    remaining = max(1, target_error - num_err)
    if extra_density is None:                             # no first extra batch yet
        trigger = n_extra >= remaining
    elif extra_density <= 0:                              # first batch found no errors -> fill a batch
        trigger = n_extra >= batch_size
    else:                                                 # queue >= remaining / density (<= one batch)
        trigger = n_extra >= min(remaining / extra_density, batch_size)
    return (trigger or flushing), flushing


class ExtraQueue:
    """Deferred-sample queue for the rebatch_speedup adaptive cap.

    The cap leaves hard (unconverged) samples behind; they are parked here on CPU
    and re-decoded later in full, uncapped batches. This class owns the offload
    scheduling (WHEN to flush, via should_flush_extra_queue) and the one-time
    hard-queue density measurement; the cap itself decides WHICH samples get
    deferred. A no-op until samples are actually deferred into it.
    """

    def __init__(self, batch_size, target_error, device='cpu'):
        self.batch_size = batch_size
        self.target_error = target_error
        self.device = device   # decoder's device: deferred samples live where they're re-decoded
        self.err = None        # [n, N] deferred errors (on self.device)
        self.llr = None        # [n, ...] deferred priors (on self.device)
        self.density = None    # errors-per-drained-sample, frozen after the FIRST flush

    def __len__(self):
        return 0 if self.err is None else self.err.shape[0]

    @property
    def nonempty(self):
        return len(self) > 0

    def defer(self, err, llr, defer_mask):
        """Park the rows selected by defer_mask (a full-batch bool tensor) onto the queue."""
        if not bool(defer_mask.any()):
            return
        d = defer_mask.to(err.device)                  # index on err's device
        e_def, l_def = err[d].detach().to(self.device), llr[d].detach().to(self.device)
        self.err = e_def if self.err is None else torch.cat((self.err, e_def))
        self.llr = l_def if self.llr is None else torch.cat((self.llr, l_def))

    def should_flush(self, num_err):
        """(do_flush, flushing): wrap should_flush_extra_queue with the queue's own state."""
        return should_flush_extra_queue(len(self), num_err, self.target_error,
                                        self.batch_size, self.density)

    def take_batch(self):
        """Pop up to batch_size of the hardest deferred samples as a one-item dataloader.
        Returns (error_dataloader, n_taken)."""
        n = min(self.batch_size, len(self))
        batch = [(self.err[:n], self.llr[:n], None)]
        self.err, self.llr = self.err[n:], self.llr[n:]
        return batch, n

    def freeze_density(self, errors_found, n_drained):
        """Set the hard-queue error density once, from the first flushed batch, then hold it."""
        if self.density is None and n_drained:
            self.density = int(errors_found) / n_drained

