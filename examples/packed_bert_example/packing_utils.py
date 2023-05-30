# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

INPUT_IDS = "input_ids"
POSITION_IDS = "position_ids"
SEGMENT_IDS = "token_type_ids"
INPUT_MASK = "attention_mask"


class DataSpec:
    def __init__(self, id, row, col, l, positions=None, flush=False, sender=0):
        self.id = id
        self.row = row
        self.col = col
        self.l = l
        self.positions = positions
        self.flush = flush
        self.sender = sender

    def __str__(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def __repr__(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def debug(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def shift(self, row_offset):
        return DataSpec(self.id, self.row - row_offset, self.col, self.l)


class DataTransfer:
    def __init__(self, count, spec, data, unpack_info, last=False):
        self.count = count
        self.specs = spec
        self.data = data
        self.unpack_info = unpack_info
        self.last = last


def insert(
    input_data,
    valid_len,
    seq_len,
    row,
    col,
    mask,
    input_ids,
    positions,
    segment_ids,
    across_rows,
    zero_delimiter_num,
):
    def pack(target_data, row, col, raw_data, valid_len, seq_len):
        if across_rows:
            if valid_len <= (seq_len - col):
                target_data[row, col : col + valid_len] = raw_data
                col += valid_len
            else:
                target_data[row, col:] = raw_data[: (seq_len - col)]
                row += 1
                target_data[row, : valid_len + col - seq_len] = raw_data[
                    seq_len - col :
                ]
                col = valid_len + col - seq_len
        else:  # user input should not across multi-rows
            if col == 0:
                target_data[row, col : col + valid_len] = raw_data
                col += valid_len
            else:
                # only available when zero_delimiter_num > 0, for deberta like network
                prepad_data = np.hstack((np.zeros(zero_delimiter_num), raw_data))
                input_len = len(prepad_data)
                if input_len <= (seq_len - col):
                    target_data[row, col : col + input_len] = prepad_data
                    col += input_len
                else:
                    row += 1
                    col = 0
                    target_data[row, col : col + valid_len] = raw_data
                    col += valid_len
        return row, col

    row_next, col_next = pack(
        input_data[INPUT_IDS], row, col, input_ids, valid_len, seq_len
    )
    pack(input_data[POSITION_IDS], row, col, positions, valid_len, seq_len)
    pack(input_data[SEGMENT_IDS], row, col, segment_ids, valid_len, seq_len)
    pack(input_data[INPUT_MASK], row, col, mask, valid_len, seq_len)
    return row_next, col_next


def create_input_data(bs, seq):
    input_data = dict()
    input_data[INPUT_IDS] = np.zeros((bs, seq), dtype=np.int32)
    input_data[POSITION_IDS] = np.zeros((bs, seq), dtype=np.int32)
    input_data[SEGMENT_IDS] = np.zeros((bs, seq), dtype=np.int32)
    input_data[INPUT_MASK] = np.zeros((bs, seq), dtype=np.int32)
    return input_data


def pack_data(
    data,
    idx,
    batch_size,
    seq_len,
    max_valid_num,
    segment_num,
    across_rows,
    zero_delimiter_num=0,
):
    input_data = create_input_data(batch_size, seq_len)
    unpack_info = np.zeros((max_valid_num, segment_num), dtype=np.int32)
    spec = []
    row, col = 0, 0
    data_num = 0

    while idx < len(data):
        features = data[idx]
        valid_len = features.input_len
        if (row == (batch_size - 1) and (seq_len - col) < valid_len) or (
            data_num >= max_valid_num
        ):
            break
        # for delimiter pack
        if (
            (row == batch_size - 1)
            and col != 0
            and (seq_len - col - zero_delimiter_num) < valid_len
        ):
            break

        unpack_info[data_num, 0] = valid_len

        input_ids = np.array(features.input_ids)
        positions = np.arange(valid_len)
        mask = np.array(features.attention_mask) + data_num
        segment_ids = np.array(features.token_type_ids)

        row, col = insert(
            input_data,
            valid_len,
            seq_len,
            row,
            col,
            mask,
            input_ids,
            positions,
            segment_ids,
            across_rows,
            zero_delimiter_num,
        )
        spec.append(DataSpec(idx, row, col, valid_len))
        data_num += 1
        idx += 1
    # mask [[1,1,1,2,0,0],[1,1,2,2,0,0]
    if across_rows == False:
        mask = input_data[INPUT_MASK]
        res = []
        for i in range(batch_size):
            begin = mask[i][0]
            start = 0
            for j in range(len(mask[i])):
                if mask[i][j] > 0:
                    mask[i][j] = mask[i][j] - begin + 1
                # for debert like network
                if zero_delimiter_num != 0:
                    if mask[i][0] == 0:
                        continue
                    if j > 0 and mask[i][j] != 0 and mask[i][j - 1] == 0:
                        res.append(j - start)
                        start = j
                    if j == len(mask[i]) - 1 and mask[i][start] != 0:
                        res.append(j - start + 1)
        for i in range(len(res)):
            unpack_info[i][0] = res[i]

    return DataTransfer(idx, spec, input_data, unpack_info)
