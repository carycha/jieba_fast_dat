import importlib.resources

# Define character type IDs
CHAR_TYPE_ZH = 0
CHAR_TYPE_NUM = 1
CHAR_TYPE_ALPHA = 2
CHAR_TYPE_OTHER = 3

# Pre-defined ranges for character types
# This map will be used to quickly determine character types
# The ranges are inclusive [start, end]
CHAR_TYPE_RANGES = [
    ((0x4E00, 0x9FA5), CHAR_TYPE_ZH),  # Chinese
    ((0x0030, 0x0039), CHAR_TYPE_NUM),  # Digits
    ((0x0041, 0x005A), CHAR_TYPE_ALPHA),  # Uppercase English
    ((0x0061, 0x007A), CHAR_TYPE_ALPHA),  # Lowercase English
]

# Pre-compute a lookup table for character types
_MAX_CHAR_CODE = 0x9FA5 + 1  # Max Chinese char code + 1
_CHAR_TYPE_LOOKUP = [CHAR_TYPE_OTHER] * _MAX_CHAR_CODE

for (start, end), char_type_id in CHAR_TYPE_RANGES:
    for char_code in range(start, end + 1):
        if char_code < _MAX_CHAR_CODE:  # Ensure we don't go out of bounds
            _CHAR_TYPE_LOOKUP[char_code] = char_type_id


def get_module_res(module, name):
    return importlib.resources.files(module).joinpath(name).open("rb")


def _get_char_type(char_code):
    if 0 <= char_code < _MAX_CHAR_CODE:
        return _CHAR_TYPE_LOOKUP[char_code]
    return CHAR_TYPE_OTHER


def split_by_char_type(text):
    if not text:
        return

    start = 0
    current_block_type = None
    n = len(text)

    for i in range(n):
        char = text[i]
        char_code = ord(char)
        char_type = _get_char_type(char_code)

        if current_block_type is None:
            current_block_type = char_type
        elif (current_block_type == CHAR_TYPE_NUM and char == ".") or (
            (
                current_block_type == CHAR_TYPE_ALPHA
                or current_block_type == CHAR_TYPE_NUM
            )
            and (
                char_type == CHAR_TYPE_ALPHA
                or char_type == CHAR_TYPE_NUM
                or char == "-"
            )
        ):
            # Continue numeric block with dot, or alpha/numeric/hyphen block
            pass
        elif char_type != current_block_type:
            # Type changed, yield previous block
            yield text[start:i], current_block_type
            start = i
            current_block_type = char_type

    # Yield the last block
    if start < n:
        yield text[start:n], current_block_type
