import numpy as np
from aocd.models import Puzzle

YEAR = 2021
DAY = 16

puzzle = Puzzle(year=YEAR, day=DAY)

# Part a
def unpack_literal(bits):
    curr_loc = 0
    literal = []
    while True:
        chunk = bits[curr_loc : curr_loc + 5]
        literal.extend(chunk[1:])
        curr_loc += 5
        if chunk[0] == 0:
            break
    itemsize = np.min_scalar_type((1 << len(literal)) - 1).itemsize
    literal = np.pad(literal, (itemsize * 8 - len(literal), 0))
    literal = np.packbits(literal[::-1], bitorder="little").view(f"<u{itemsize}")[0]
    return literal, curr_loc


def parse_packet(bits):
    curr_loc = 0
    s = 0
    p_version = bits[curr_loc : curr_loc + 3]
    p_type = bits[curr_loc + 3 : curr_loc + 6]
    p_version = np.packbits(p_version[::-1], bitorder="little")[0]
    s += p_version
    p_type = np.packbits(p_type[::-1], bitorder="little")[0]
    curr_loc += 6
    if p_type == 4:
        value, nbr_bits = unpack_literal(bits[curr_loc:])
        curr_loc += nbr_bits
    else:
        l_type = bits[curr_loc]
        curr_loc += 1
        sub_packet_values = []
        if l_type == 0:
            sub_packet_bits = bits[curr_loc : curr_loc + 15]
            sub_packet_bits = np.packbits(sub_packet_bits[::-1], bitorder="little").view("<u2")[0]
            curr_loc += 15
            n = 0
            while n < sub_packet_bits:
                e, nbr_bits, v = parse_packet(bits[curr_loc:])
                s += e
                n += nbr_bits
                curr_loc += nbr_bits
                sub_packet_values.append(v)
            assert n == sub_packet_bits
        else:
            nbr_sub_packets = bits[curr_loc : curr_loc + 11]
            nbr_sub_packets = np.packbits(
                nbr_sub_packets[::-1], bitorder="little"
            ).view("<u2")[0]
            curr_loc += 11
            for _ in range(nbr_sub_packets):
                e, nbr_bits, v = parse_packet(bits[curr_loc:])
                s += e
                curr_loc += nbr_bits
                sub_packet_values.append(v)
        ops = {
            0: np.sum,
            1: np.prod,
            2: np.min,
            3: np.max,
            5: np.greater,
            6: np.less,
            7: np.equal,
        }
        if p_type < 4:
            value = int(ops[p_type](sub_packet_values))
        else:
            value = int(ops[p_type](*sub_packet_values))
    return s, curr_loc, value


def a(data):
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(data), dtype=np.uint8))
    s = 0
    while len(bits) > 7:
        e, nbr_bits, _ = parse_packet(bits)
        bits = bits[nbr_bits:]
        s += e
    return s


# a("D2FE28")
# a("38006F45291200")
# a("EE00D40C823060")

assert a("8A004A801A8002F478") == 16
assert a("620080001611562C8802118E34") == 12
assert a("C0015000016115A2E0802F182340") == 23
assert a("A0016C880162017C3686B18A3D4780") == 31
answer = a(puzzle.input_data)
print("a:", answer)
assert answer == 920


# Part b
def b(data):
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(data), dtype=np.uint8))
    return parse_packet(bits)[-1]


assert b("C200B40A82") == 3
assert b("04005AC33890") == 54
assert b("880086C3E88112") == 7
assert b("CE00C43D881120") == 9
assert b("D8005AC2A8F0") == 1
assert b("F600BC2D8F") == 0
assert b("9C005AC2F8F0") == 0
assert b("9C0141080250320F1802104A08") == 1
answer = b(puzzle.input_data)
print("b:", answer)
assert answer == 10185143721112
