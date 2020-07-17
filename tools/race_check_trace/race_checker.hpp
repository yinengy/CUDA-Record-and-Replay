/* 
 * a data race checker
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */
 
#include <unordered_set>
#include <unordered_map>
#include <stdint.h>

#include "common.h"

class Checker {
private:
    struct Instruction {
        int func_id;
        int inst_id;

        Instruction(int func_id, int inst_id) {
            func_id = func_id;
            inst_id = inst_id;
        }

        bool operator==(const Instruction &other) const { 
            return (func_id == other.func_id &&
                    inst_id == other.inst_id);
        }
    };

    struct HashInstruction { 
        size_t operator()(const Instruction &i) const
        { 
            return (std::hash<int>()(i.func_id)) ^  
                (std::hash<int>()(i.inst_id)); 
        } 
    }; 

    struct Address {
        std::unordered_set<int> load;   // set of Thread ID(or Block ID for inter block races) that read from this address
        std::unordered_set<int> store;  // set of Thread ID(or Block ID for inter block races) that write to this address
        std::unordered_set<Instruction, HashInstruction> insts;  // set of Inst wrt to Thread ID in self.load and self.store

        Address() {
            load = std::unordered_set<int>();
            store = std::unordered_set<int>();
            insts = std::unordered_set<Instruction, HashInstruction>();
        }
    };

    // SFR in a block
    struct SFR {
        int block_id;
        int SFR_id;

        SFR(int block_id, int SFR_id) {
            this->block_id = block_id;
            this->SFR_id = SFR_id;
        }

        bool operator==(const SFR &other) const { 
            return (block_id == other.block_id &&
                    SFR_id == other.SFR_id);
        }
    };

    struct HashSFR { 
        size_t operator()(const SFR &i) const
        { 
            return (std::hash<int>()(i.block_id)) ^  
                (std::hash<int>()(i.SFR_id)); 
        } 
    }; 

    // key: SFR, val: shared_mem (a dic of addr : Address (has two set of Thread ID))
    std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR> SFR_shared_mem;

    // key: SFR, val: global_mem (a dic of addr : Address (has two set of Thread ID))
    std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR> SFR_global_mem;

    // key: addr, val: Address (has two set of Block ID)
    std::unordered_map<uint64_t, Address> Global_mem;

public:
    Checker() {
        SFR_shared_mem = std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR>();
        SFR_global_mem = std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR>();
        Global_mem = std::unordered_map<uint64_t, Address>();
    }

    // read input which is the memory access information
    void read(mem_access_t *ma) {
        SFR s = SFR(ma->block_id, ma->SFR_id);
        int base_thread_id = ma->warp_id * 32;
        Instruction inst = Instruction(ma->func_id, ma->inst_id);

        for (int i = 0; i < 32; i++) {
            uint64_t addr = ma->addrs[i];
            int thread_id = base_thread_id + i;

            // TODO: check active mask rather than skip addr == 0
            if (addr == 0) {
                continue;
            }

            if (ma->is_load) {
                if (ma->is_shared_memory) {
                    Address a = SFR_shared_mem_get(s, addr);
                    a.load.insert(thread_id);
                    a.insts.insert(inst);
                } else { // global memory
                    // intra block
                    Address a = SFR_global_mem_get(s, addr);
                    
                    a.load.insert(thread_id);
                    a.insts.insert(inst);

                    // inter block
                    Address g = Global_mem_get(addr);

                    // add block id rather than thread id
                    g.load.insert(ma->block_id);
                    g.insts.insert(inst);
                }
            } else {
                if (ma->is_shared_memory) {
                    Address a = SFR_shared_mem_get(s, addr);
                    a.store.insert(thread_id);
                    a.insts.insert(inst);
                } else { // global memory
                    // intra block
                    Address a = SFR_global_mem_get(s, addr);
                    
                    a.store.insert(thread_id);
                    a.insts.insert(inst);

                    // inter block
                    Address g = Global_mem_get(addr);

                    // add block id rather than thread id
                    g.store.insert(ma->block_id);
                    g.insts.insert(inst);
                }
            }
        }   
    }

    // check data race based on current information
    // will print "func_id,inst_id" involved
    void check() {
        return;
    }

private:
    // if s or addr doesn't exits, insert
    // return 
    Address SFR_shared_mem_get(SFR s, uint64_t addr) {
        auto iter1 = SFR_shared_mem.find(s);
        std::unordered_map<uint64_t, Address> shared_mem;
        if (iter1 == SFR_shared_mem.end()) {
            shared_mem = std::unordered_map<uint64_t, Address>();
            SFR_shared_mem[s] = shared_mem;
        } else {
            shared_mem = iter1->second;
        }

        auto iter2 = shared_mem.find(addr);
        Address a;
        if (iter2 == shared_mem.end()) {
            a = Address();
            shared_mem[addr] = a;
        } else {
            a = iter2->second;
        }

        return a;
    }

    // if s or addr doesn't exits, insert
    Address SFR_global_mem_get(SFR s, uint64_t addr) {
        auto iter1 = SFR_global_mem.find(s);
        std::unordered_map<uint64_t, Address> global_mem;
        if (iter1 == SFR_global_mem.end()) {
            global_mem = std::unordered_map<uint64_t, Address>();
            SFR_global_mem[s] = global_mem;
        } else {
            global_mem = iter1->second;
        }

        auto iter2 = global_mem.find(addr);
        Address a;
        if (iter2 == global_mem.end()) {
            a = Address();
            global_mem[addr] = a;
        } else {
            a = iter2->second;
        }

        return a;
    }

    // if addr doesn't exits, insert
    Address Global_mem_get(uint64_t addr) {
        auto iter = Global_mem.find(addr);
        Address g;
        if (iter == Global_mem.end()) {
            g = Address();
            Global_mem[addr] = g;
        } else {
            g = iter->second;
        }

        return g;
    }
};