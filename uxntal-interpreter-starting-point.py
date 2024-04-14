#! Comments of this style (`#!` or `#!!`) indicate code that needs completing or changing
# ! A `#!!` means "must"
# ! "should" or "must" means you'll lose marks if you don't do it
# ! "could" or "(optional)" means you can get extra marks for doing it, but you don't lose marks for not doing it

from enum import Enum

# ! Your program could set these flags on command line
V = False  # Verbose, explain a bit what happens
VV = False  # More verbose, explain in more detail what happens
DBG = False  # Debug info

# !! Your program should read the program text from a `.tal` file
programText = """
|0100
;array LDA
;array #01 ADD LDA
ADD
;print JSR2
BRK

@print
    #18 DEO
JMP2r

@array 11 22 33
"""


# These are the different types of tokens
class T(Enum):
    MAIN = 0  # Main program
    LIT = 1  # Literal
    INSTR = 2  # Instruction
    LABEL = 3  # Label
    REF = 4  # Address reference (rel=1, abs=2)
    RAW = 5  # Raw values (i.e. not literal)
    ADDR = 6  # Address (absolute padding)
    PAD = 7  # Relative padding
    EMPTY = 8  # Memory is filled with this by default


# We use an object to group the data structures used by the Uxn interpreter
class UxnMachine:
    memory = [(T.EMPTY,)] * 0x10000  # The memory stores *tokens*, not bare values
    stacks = ([], [])  # ws, rs # The stacks store bare values, i.e. bytes
    progCounter = 0
    symbolTable = {}
    # First unused address, only used for verbose
    free = 0


# !! Complete the parser
def parse_token(token_str):
    if token_str[0] == '#':
        val_str = token_str[1:]
        val = int(val_str, 16)
        if len(val_str) == 2:
            return T.LIT, val, 1
        else:
            return T.LIT, val, 2
    if token_str[0] == '"':
        chars = list(token_str[1:])
        return list(map(lambda c: (T.LIT, ord(c), 1), chars))
    elif token_str[0] == ';':
        val = token_str[1:]
        return T.REF, val, 2
    # !! Handle relative references `,&`
    elif token_str[0:2] == ',&':
        val = token_str[2:]
        return T.REF, val, 1
    elif token_str[0] == '@':
        val = token_str[1:]
        return T.LABEL, val
    # !! Handle relative labels `&`
    elif token_str[0] == '&':
        val = token_str[1:]
        return T.LABEL, val
    elif token_str == '|0100':
        return (T.MAIN,)
    # ! Handle absolute padding (optional)
    # elif tokenStr[0] == ...
    #     ...
    #     return (...)
    # ! Handle relative padding
    elif token_str[0] == '$':
        val = token_str[1:]
        return T.PAD, val
    elif token_str[0].isupper():
        # Any token string starting with an uppercase letter is considered an instruction
        if len(token_str) == 3:
            return T.INSTR, token_str[0:len(token_str)], 1, 0, 0
        elif len(token_str) == 4:
            if token_str[-1] == '2':
                return T.INSTR, token_str[0:len(token_str) - 1], 2, 0, 0
            elif token_str[-1] == 'r':
                return T.INSTR, token_str[0:len(token_str) - 1], 1, 1, 0
            elif token_str[-1] == 'k':
                return T.INSTR, token_str[0:len(token_str) - 1], 1, 0, 1
        elif len(token_str) == 5:
            # Order must be size:stack:keep
            if token_str[len(token_str) - 2:len(token_str)] == '2r':
                return T.INSTR, token_str[0:len(token_str) - 2], 2, 1, 0
            elif token_str[len(token_str) - 2:len(token_str)] == '2k':
                return T.INSTR, token_str[0:len(token_str) - 2], 2, 0, 1
            elif token_str[len(token_str) - 2:len(token_str)] == 'rk':
                return T.INSTR, token_str[0:len(token_str) - 2], 1, 1, 1
        elif len(token_str) == 6:
            return T.INSTR, token_str[0:len(token_str) - 1], 2, 1, 1
    else:
        # we assume this is a 'raw' byte or short
        return T.RAW, int(token_str, 16)


# These are the actions related to the various Uxn instructions

# Memory operations
# STA
def store(args, sz, uxn):
    uxn.memory[args[1]] = ('RAW', args[0], 0)


# LDA
def load(args, sz, uxn):
    return uxn.memory[args[0]][1]  # memory has tokens, stacks have values


# Control operations
# JSR
def call(args, sz, uxn):
    # print("CALL:",args[0],uxn.progCounter)
    uxn.stacks[1].append(uxn.progCounter)
    uxn.progCounter = args[0] - 1


# JMP
def jump(args, sz, uxn):
    uxn.progCounter = args[0]


# JCN
def cond_jump(args, sz, uxn):
    if args[0] == 1:
        uxn.progCounter = args[0] - 1


# Stack manipulation operations
# STH
def stash(rs, sz, uxn):
    uxn.stacks[1 - rs].append(uxn.stacks[rs].pop())


# !! Implement POP (look at `swap`)
def pop(rs, sz, uxn):
    uxn.stacks[rs].pop()


# SWP
def swap(rs, sz, uxn):
    b = uxn.stacks[rs].pop()
    a = uxn.stacks[rs].pop()
    uxn.stacks[rs].append(b)
    uxn.stacks[rs].append(a)


# This implementation of NIP check if the words on the stack match the mode (short or byte)
# ! Your implementations of the other stack operations don't need to do this
def nip(rs, sz, uxn):  # a b -> b
    b = uxn.stacks[rs].pop()
    if b[1] == sz:
        a = uxn.stacks[rs].pop()
        if a[1] == sz:
            uxn.stacks[rs].append(b)
        else:
            print("Error: Args on stack for NIP", sz, "are of wrong size")
            exit()
    elif b[1] == 2 and sz == 1:
        bb = b[0] & 0xFF
        uxn.stacks[rs].append((bb, 1))
    elif b[1] == 1 and sz == 2:
        print("Error: Args on stack for NIP", sz, "are of wrong size")
        exit()


# !! Implement ROT (look at `swap`)
def rot(rs, sz, uxn):  # a b c -> b c a
    c = uxn.stacks[rs].pop()
    b = uxn.stacks[rs].pop()
    a = uxn.stacks[rs].pop()
    uxn.stacks[rs].append(b)
    uxn.stacks[rs].append(c)
    uxn.stacks[rs].append(a)


def dup(rs, sz, uxn):
    a = uxn.stacks[rs][-1]
    uxn.stacks[rs].append(a)


def over(rs, sz, uxn):  # a b -> a b a
    a = uxn.stacks[rs][-2]
    uxn.stacks[rs].append(a)


# ALU operations TODO assumption that args[0] is lowest on stack
# ADD
def add(args, sz, uxn):
    return args[0] + args[1]


# !! Implement SUB, MUL, DIV, INC (similar to `ADD`)
def sub(args, sz, uxn):
    return args[0] - args[1]


def mul(args, sz, uxn):
    return args[0] * args[1]


def div(args, sz, uxn):
    return args[0] // args[1]


def inc(rs, sz, uxn):  # TODO this isn't how its suggested to implement it
    uxn.stacks[rs] += 1


# !! Implement EQU, NEQ, LTH, GTH (similar to `ADD`)
def equ(args, sz, uxn):
    return args[0] == args[1]


def neq(args, sz, uxn):
    return args[0] != args[1]


def lth(args, sz, uxn):
    return args[0] < args[1]


def gth(args, sz, uxn):
    return args[0] > args[1]


callInstr = {
    # !! Add SUB, MUL, DIV, INC; EQU, NEQ, LTH, GTH
    'ADD': (add, 2, True),
    'SUB': (sub, 2, True),
    'MUL': (mul, 2, True),
    'DIV': (div, 2, True),
    'INC': (inc, 0, False),
    'EQU': (equ, 2, True),
    'NEQ': (neq, 2, True),
    'LTH': (lth, 2, True),
    'GTH': (gth, 2, True),
    'DEO': (lambda args, sz, uxn: print(chr(args[1]), end=''), 2, False),
    'JSR': (call, 1, False),
    'JMP': (jump, 1, False),
    'JCN': (cond_jump, 2, False),
    'LDA': (load, 1, True),
    'STA': (store, 2, False),
    'STH': (stash, 0, False),
    'DUP': (dup, 0, False),
    'SWP': (swap, 0, False),
    'OVR': (over, 0, False),
    'NIP': (nip, 0, False),
    # !! Add POP, ROT
    'POP': (nip, 0, False),
    'ROT': (rot, 0, False),
}


def execute_instr(instr_token, uxn):
    _t, instr, sz, rs, keep = instr_token
    if instr == 'BRK':
        if V:
            print("\n", '*** DONE *** ')
        else:
            print('')
        if VV:
            print('PC:', uxn.pc, ' (WS,RS):', uxn.stacks)
        exit(0)
    action, n_args, has_res = callInstr[instr]
    if n_args == 0:  # means it is a stack manipulation
        action(rs, sz, uxn)
    else:
        # args=[]
        # for i in reversed(range(0,n_args)):
        #     if keep == 0:
        #         args.append(uxn.stacks[rs].pop())
        #     else:
        #         args.append(uxn.stacks[rs][i])
        args = []
        for i in reversed(range(0, n_args)):
            if keep == 0:
                arg = uxn.stacks[rs].pop()
                if arg[1] == 2 and sz == 1 and (instr != 'LDA' and instr != 'STA'):
                    if VV:
                        print("Warning: Args on stack for", instr, sz, "are of wrong size (short for byte)")
                    uxn.stacks[rs].append((arg[0] >> 8))
                    args.append((arg[0] & 0xFF))
                else:  # either 2 2 or 1 1 or 1 2
                    args.append(arg[0])  # works for 1 1 or 2 2
                    if arg[1] == 1 and sz == 2:
                        arg1 = arg
                        arg2 = uxn.stacks[rs].pop()
                        if arg2[1] == 1 and sz == 2:
                            arg = (arg2[0] << 8) + arg1[0]
                            args.append(arg)  # a b
                        else:
                            print("Error: Args on stack are of wrong size (short after byte)")
                            exit()
            else:
                arg = uxn.stacks[rs][i]
                if arg[1] != sz and (instr != 'LDA' and instr != 'STA'):
                    print("Error: Args on stack are of wrong size (keep)")
                    exit()
                else:
                    args.append(arg[0])
        if VV:
            print('EXEC INSTR:', instr, 'with args', args)
        if has_res:
            res = action(args, sz, uxn)
            if instr == 'EQU' or instr == 'NEQ' or instr == 'LTH' or instr == 'GTH':
                uxn.stacks[rs].append((res, 1))
            else:
                uxn.stacks[rs].append((res, sz))
        else:
            action(args, sz, uxn)


# !! Tokenize the program text using a function `tokenizeProgramText`
# ! That means splitting the string `programText` on whitespace
# ! You must remove any comments first, I suggest you use a helper function stripComments
def find_all(a_str, sub_str) -> list:
    occurrences = []  # will hold index of start of all occurrences

    found_all = False
    start = 0
    while not found_all:
        start = a_str.find(sub_str, start)  # returns -1 if not found
        if start == -1:
            found_all = True
        else:
            occurrences.append(start)
        start += len(sub_str)
    return occurrences


def strip_comments(
        program_text):  # find the index of every open and close bracket, then remove the space between them (inclusive)
    stripped = ""
    # find location of all brackets
    comment_starts = find_all(program_text, "(") + [len(program_text) - 1]
    comment_ends = [0] + find_all(program_text, ")")

    # remove content between them (inclusive)
    for index in range(len(comment_starts)):
        start = comment_ends[index]
        end = comment_starts[index]
        stripped += program_text[start + 1:end]
    return stripped


# ! `tokenStrings` is a list of all tokens as strings
def tokenize_program_text(program_text):
    # ! ...
    # prepare for tokenization
    token_strings = program_text.split(" ")

    # tokenize

    return []  # ! replace this with the actual code


# This is the first pass of the assembly process
# We store the tokens in memory and build a dictionary
# uxn.symbolTable: label => address
def populate_memory_and_build_symbol_table(token_list, uxn):
    prog_counter = 0
    for token_item in token_list:
        if token_item == (T.MAIN,):
            prog_counter = 0x0100
        elif token_item[0] == T.ADDR:
            prog_counter = token_item[1]
        elif token_item[0] == T.PAD:  # relative only
            prog_counter = prog_counter + token_item[1]
        elif token_item[0] == T.LABEL:
            label_name = token_item[1]
            uxn.symbolTable[label_name] = prog_counter
        else:
            uxn.memory[prog_counter] = token_item
            prog_counter = prog_counter + 1
    uxn.free = prog_counter


# Once the symbol table has been built, replace every symbol by its address !! Implement the code to replace every
# label reference by an address ! Note that label references are `REF` tokens and the memory stores the symbolTable
# as `LIT` tokens ! Loop over all tokens in `uxn.memory``. If a token is `REF`, look it up in `uxn.symbolTable`` and
# create a `LIT` token that contains its address. Write that to the memory. ! (This is what happens in Uxn: `;label`
# is the same as `LIT2 =label` and that gets replaced by `LIT2 address`)

# ! def resolveSymbols(uxn):
# !  ...

# Running the program mean setting the program counter `uxn.progCounter` to the address of the first token;
#  - read the token from memory at that address
# - if the token is a LIT, its *value* goes on the working stack
# - otherwise it is an instruction, and it is executed using `executeInstr(token,uxn)`
# - then we increment the program counter
# !! Implement the above functionality
def run_program(uxn):
    if VV:
        print('*** RUNNING ***')
    uxn.progCounter = 0x100  # all programs must start at 0x100
    while True:
        # !! read the token from memory at that address
        # ! token = ...
        if DBG:
            print('PC:', uxn.progCounter, ' TOKEN:', token)
        # ! You can use an if/elif if you prefer; there are only two cases
        # ! (and an optional third to catch potential errors)
        # ! because the program at this point consists entirely of instructions and literals
        # ! match ...:
        # !     case ...:
        # !         ...
        # !     case ...:
        # !         ...
        # !! Increment the program counter
        # ! ...
        if DBG:
            print('(WS,RS):', uxn.stacks)


uxnMachine = UxnMachine()
programText_noComments = strip_comments(programText)
tokenStrings = tokenize_program_text(programText_noComments)
tokensWithStrings = map(parse_token, tokenStrings)

tokens = []
for item in tokensWithStrings:
    if type(item) == list:
        for token in item:
            tokens.append(token)
    else:
        tokens.append(item)

populate_memory_and_build_symbol_table(tokens, uxnMachine)

resolveSymbols(uxnMachine)

if DBG:
    for pc in range(256, uxnMachine.free):
        print(pc, ':', uxnMachine.memory[pc])
    print('')
if VV:
    print(programText)

run_program(uxnMachine)
