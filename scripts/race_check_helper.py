#!/usr/bin/env python3
#
# The script will read stdin 
# (which should be output of NVbit tool "race_check_trace").
# And it will grep load and store. 
# Will check if there are conflicting memory accesses (data races)
# A warning will be printed
#
# the difference of this script and race_check_helper_memaddr.py is that
# race_check_helper_memaddr.py reports data race with respect to memory address
# while this script reports data race with respect to instructions
#
# Yineng Yan (yinengy@umich.edu), 2020

import sys

kernel_counter = 0

functions = []

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Address:
    def __init__(self):
        self.load = set() # set of Thread(or Block for inter block races) that read from this address
        self.store = set() # set of Thread(or Block for inter block races) that write to this address
        self.insts = set() # set of Inst wrt to Thread in self.load and self.store


class instruction:
    def __init__(self, func_id, inst_id):
        self.func_id = func_id
        self.inst_id = inst_id
    
    def __hash__(self):
        return hash((self.func_id, self.inst_id))
    
    def __eq__(self, other):
        return self.func_id == other.func_id and \
        self.inst_id == other.inst_id

    def __str__(self):
        # at function name, instruction SASS)
        return "{},{}".format(self.func_id, self.inst_id)


# Thread in a block
class Thread:
    def __init__(self, warp_id, lane_id):
        self.warp_id = warp_id
        self.lane_id = lane_id

    def __hash__(self):
         return hash((self.warp_id, self.lane_id))
    
    def __eq__(self, other):
        return self.warp_id == other.warp_id and \
        self.lane_id == other.lane_id

    def __str__(self):
        # (warp ID, lane ID)
        return "({} {})".format(self.warp_id, self.lane_id)


# SFR in a block
class SFR:
    def __init__(self, cta_id_x, cta_id_y, cta_id_z, SFR_id):
        self.cta_id_x = cta_id_x
        self.cta_id_y = cta_id_y
        self.cta_id_z = cta_id_z
        self.SFR_id = SFR_id

    def __hash__(self):
         return hash((self.cta_id_x, self.cta_id_y, self.cta_id_z, self.SFR_id))
    
    def __eq__(self, other):
        return self.cta_id_x == other.cta_id_x and \
        self.cta_id_y == other.cta_id_y and \
        self.cta_id_z == other.cta_id_z and \
        self.SFR_id == other.SFR_id

    def __str__(self):
        # "Block_id: (CTA.x CTA.y CTA.z), SFR_id: SFR ID"
        return "Block_id: ({} {} {}), SFR_id: {}".format(self.cta_id_x, self.cta_id_y, self.cta_id_z, self.SFR_id)


class Block:
    def __init__(self, cta_id_x, cta_id_y, cta_id_z):
        self.cta_id_x = cta_id_x
        self.cta_id_y = cta_id_y
        self.cta_id_z = cta_id_z

    def __hash__(self):
         return hash((self.cta_id_x, self.cta_id_y, self.cta_id_z))
    
    def __eq__(self, other):
        return self.cta_id_x == other.cta_id_x and \
        self.cta_id_y == other.cta_id_y and \
        self.cta_id_z == other.cta_id_z

    def __str__(self):
        # (CTA.x CTA.y CTA.z)
        return "({} {} {})".format(self.cta_id_x, self.cta_id_y, self.cta_id_z)


class Function:
    def __init__(self, func_name):
        self.func_name = func_name
        self.insts = []


def process_message():
    global functions

    SFR_shared_mem = {} # key: SFR, val: shared_mem (a dic of addr : Address (has two set of Thread))
    SFR_global_mem = {} # key: SFR, val: global_mem (a dic of addr : Address (has two set of Thread))

    GLOBAL_mem = {} # key: addr, val: Address (has two set of Block)

    # flag and counter for reading function assembly
    read_func = False
    read_func_name = False

    # read input and build dict
    for line in sys.stdin:
        # handle special message (kernel ends signal and function assembly)
        if line[0] != '#': # all message begin with #
            continue
        elif line.strip('\n') == "#kernelends#":
            check_result(SFR_shared_mem, SFR_global_mem, GLOBAL_mem)
            # do a new loop
            SFR_shared_mem = {} # key: SFR, val: shared_mem (a dic of addr : Address)
            SFR_global_mem = {} # key: SFR, val: global_mem (a dic of addr : Address)

            GLOBAL_mem = {} # key: addr, val: Address (a set of Block)
            continue
        elif line[:12] == "#func_begin#": # begins reading functions
            read_func = True
            functions.append(Function(line.strip('\n')[12:]))
            continue
        elif line.strip('\n') == "#func_end#": # finish reading functions
            read_func = False
            continue
        elif read_func and line[:6] == "#SASS#":
            functions[-1].insts.append(line.strip('\n')[6:])
            continue
        
        # handle load and store message
        # format: "#ld#is_shared_memory, cta_id_x, cta_id_y, cta_id_z, warp_id, lane_id, func_id, inst_id, SFR_id, addr\n"
        temp = line.strip('\n')[4:].split(",")

        if (len(temp) != 10):  # skip unwanted output
            continue

        addr = temp[-1]
        t = Thread(temp[4], temp[5])
        s = SFR(temp[1], temp[2], temp[3], temp[-2])
        b = Block(temp[1], temp[2], temp[3])
        inst = instruction(int(temp[6]), int(temp[7]))

        if "#ld#" in line: 
            if (temp[0] == '1'): # shared memory
                if s not in SFR_shared_mem:
                    SFR_shared_mem[s] = {}

                shared_mem = SFR_shared_mem[s]

                if addr not in shared_mem:
                    shared_mem[addr] = Address()
                shared_mem[addr].load.add(t) 
                # add inst to inst set
                shared_mem[addr].insts.add(inst)

            else: # global memory
                # intra block
                if s not in SFR_global_mem:
                    SFR_global_mem[s] = {}

                global_mem = SFR_global_mem[s]

                if addr not in global_mem:
                    global_mem[addr] = Address()

                global_mem[addr].load.add(t) 
                # add inst to inst set
                global_mem[addr].insts.add(inst)    

                # inter block
                if addr not in GLOBAL_mem:
                    GLOBAL_mem[addr] = Address()

                GLOBAL_mem[addr].load.add(b)  # add block rather than thread
                # add inst to inst set
                GLOBAL_mem[addr].insts.add(inst) 

        elif "#st#" in line: # format: "#st#is_shared_memory, cta_id_x, cta_id_y, cta_id_z, warp_id, lane_id,func_id, inst_id, SFR_id, addr\n"
            if (temp[0] == '1'): # shared memory
                if s not in SFR_shared_mem:
                    SFR_shared_mem[s] = {}

                shared_mem = SFR_shared_mem[s]

                if addr not in shared_mem:
                    shared_mem[addr] = Address()
                shared_mem[addr].store.add(t) 
                # add inst to inst set
                shared_mem[addr].insts.add(inst)
            else:
                # intra block
                if s not in SFR_global_mem:
                    SFR_global_mem[s] = {}

                global_mem = SFR_global_mem[s]

                if addr not in global_mem:
                    global_mem[addr] = Address()

                global_mem[addr].store.add(t)  
                # add inst to inst set
                global_mem[addr].insts.add(inst)  

                # inter block
                if addr not in GLOBAL_mem:
                    GLOBAL_mem[addr] = Address()

                GLOBAL_mem[addr].store.add(b)  # add block rather than thread
                # add inst to inst set
                GLOBAL_mem[addr].insts.add(inst) 


def check_result(SFR_shared_mem, SFR_global_mem, GLOBAL_mem):
    global kernel_counter
    kernel_counter += 1

    inter_block_global_memory_counter = 0

    # intra block shared memory
    intra_shared_races = set()

    for SFR, shared_mem in SFR_shared_mem.items():
        for addr, addr_obj in shared_mem.items():
            if (len(addr_obj.store) > 1) or (len(addr_obj.store) == 1 and \
                (len(addr_obj.load) >= 1 and addr_obj.load != addr_obj.store)) :
                intra_shared_races.add(frozenset(addr_obj.insts))

    # report races
    for race in intra_shared_races:
        for inst in race:
            print(inst)
                

    # intra block global memory
    intra_global_races = set()
    for SFR, global_mem in SFR_global_mem.items():
        for addr, addr_obj in global_mem.items():
            if (len(addr_obj.store) > 1) or (len(addr_obj.store) == 1 and \
                (len(addr_obj.load) >= 1 and addr_obj.load != addr_obj.store)) :
                intra_global_races.add(frozenset(addr_obj.insts))

    # report races
    for race in intra_global_races:
        for inst in race:
            print(inst)

    # inter block global memory
    inter_global_races = set()
    for addr, addr_obj in GLOBAL_mem.items():
        if (len(addr_obj.store) > 1) or (len(addr_obj.store) == 1 and \
            (len(addr_obj.load) >= 1 and addr_obj.load != addr_obj.store)) :
            inter_global_races.add(frozenset(addr_obj.insts))

    # report races
    for race in inter_global_races:
        for inst in race:
            print(inst)

if __name__ == "__main__":
    process_message()