/* 
 * a data race checker
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */
 
#include <google/dense_hash_set>
#include <google/dense_hash_map>
#include <stdint.h>
#include <algorithm> // set_union
#include <iostream>
#include <fstream>

#include "common.h"


class Checker {
private:
    struct Instruction {
        int func_id;
        int inst_id;

        Instruction() { }

        Instruction(int func_id, int inst_id) {
            this->func_id = func_id;
            this->inst_id = inst_id;
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
        google::dense_hash_set<int> load;   // set of Thread ID(or Block ID for inter block races) that read from this address
        google::dense_hash_set<int> store;  // set of Thread ID(or Block ID for inter block races) that write to this address
        google::dense_hash_set<Instruction, HashInstruction> insts;  // set of Inst wrt to Thread ID in self.load and self.store

        Address() {
            load = google::dense_hash_set<int>();
            store = google::dense_hash_set<int>();
            insts = google::dense_hash_set<Instruction, HashInstruction>();

            load.set_empty_key(-1);
            store.set_empty_key(-1);
            insts.set_empty_key(Instruction(-1, -1));
        }

        // memory accesses from different thread and at least one store
        bool has_race() {
            if (store.size() > 1) {
                return true;
            } else if (store.size() == 1) {
                if (load.size() == 1) {
                    // load and store from the same thread
                    return load != store;
                } else {
                    return true;
                }
            } 
            
            return false; // no store
        }
    };

    // SFR in a block
    struct SFR {
        int block_id;
        int SFR_id;

        SFR() {}

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
    google::dense_hash_map<SFR, google::dense_hash_map<uint64_t, Address>, HashSFR> SFR_shared_mem;

    // key: SFR, val: global_mem (a dic of addr : Address (has two set of Thread ID))
    google::dense_hash_map<SFR, google::dense_hash_map<uint64_t, Address>, HashSFR> SFR_global_mem;

    // key: addr, val: Address (has two set of Block ID)
    google::dense_hash_map<uint64_t, Address> Global_mem;

public:
    Checker() {
        SFR_shared_mem = google::dense_hash_map<SFR, google::dense_hash_map<uint64_t, Address>, HashSFR>();
        SFR_global_mem = google::dense_hash_map<SFR, google::dense_hash_map<uint64_t, Address>, HashSFR>();
        Global_mem = google::dense_hash_map<uint64_t, Address>();

        /* required by google dense_hash_map */
        SFR_shared_mem.set_empty_key(SFR(-1, -1));
        SFR_global_mem.set_empty_key(SFR(-1, -1));
        Global_mem.set_empty_key(0);
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
                    Address &a = SFR_shared_mem_get(s, addr);
                    a.load.insert(thread_id);
                    a.insts.insert(inst);
                } else { // global memory
                    // intra block
                    Address &a = SFR_global_mem_get(s, addr);
                    
                    a.load.insert(thread_id);
                    a.insts.insert(inst);

                    // inter block
                    Address &g = Global_mem_get(addr);

                    // add block id rather than thread id
                    g.load.insert(ma->block_id);
                    g.insts.insert(inst);
                }
            } else {
                if (ma->is_shared_memory) {
                    Address &a = SFR_shared_mem_get(s, addr);
                    a.store.insert(thread_id);
                    a.insts.insert(inst);
                } else { // global memory
                    // intra block
                    Address &a = SFR_global_mem_get(s, addr);
                    
                    a.store.insert(thread_id);
                    a.insts.insert(inst);

                    // inter block
                    Address &g = Global_mem_get(addr);

                    // add block id rather than thread id
                    g.store.insert(ma->block_id);
                    g.insts.insert(inst);
                }
            }
        }   
    }

    // check data race based on current information
    // will print "func_id,inst_id" involved
    void check(std::ofstream & output_file) {
        auto races = google::dense_hash_set<Instruction, HashInstruction>();
        races.set_empty_key(Instruction(-1, -1));

        // intra block shared memory
        for (auto& itr1 : SFR_shared_mem) {
            auto shared_mem = itr1.second;
            for (auto& itr2 : shared_mem) {
                Address a = itr2.second;
                if (a.has_race()) {
                    races.insert(a.insts.begin(), a.insts.end());
                }
            }
        }

        // intra block global memory
        for (auto& itr1 : SFR_global_mem) {
            auto global_mem = itr1.second;
            for (auto& itr2 : global_mem) {
                Address a = itr2.second;
                if (a.has_race()) {
                    races.insert(a.insts.begin(), a.insts.end());
                }
            }
        }

        // inter block global memory
        for (auto& itr : Global_mem) {
            auto a = itr.second;
            if (a.has_race()) {
                races.insert(a.insts.begin(), a.insts.end());
            }
        }

        // report races
        for (auto inst : races) {
            output_file << inst.func_id << "," << inst.inst_id << "\n";
        }

        return;
    }

private:
    // if s or addr doesn't exits, insert
    Address & SFR_shared_mem_get(SFR s, uint64_t addr) {
        google::dense_hash_map<uint64_t, Address> shared_mem;
        shared_mem.set_empty_key(0);
        if (SFR_shared_mem.find(s) == SFR_shared_mem.end()) {
            google::dense_hash_map<uint64_t, Address> temp;
            temp.set_empty_key(0);
            SFR_shared_mem[s] = temp; 
        } 

        shared_mem = SFR_shared_mem[s];

        if (shared_mem.find(addr) == shared_mem.end()) {
            shared_mem[addr] = Address();
        } 

        return SFR_shared_mem[s][addr];
    }

    // if s or addr doesn't exits, insert
    Address & SFR_global_mem_get(SFR s, uint64_t addr) {
        google::dense_hash_map<uint64_t, Address> global_mem;
        global_mem.set_empty_key(0);
        if (SFR_global_mem.find(s) == SFR_global_mem.end()) {
            google::dense_hash_map<uint64_t, Address> temp;
            temp.set_empty_key(0);
            SFR_global_mem[s] = temp; 
        } 

        global_mem = SFR_global_mem[s];

        if (global_mem.find(addr) == global_mem.end()) {
            global_mem[addr] = Address();
        } 

        return SFR_global_mem[s][addr];
    }

    // if addr doesn't exits, insert
    Address & Global_mem_get(uint64_t addr) {
        if (Global_mem.find(addr) == Global_mem.end()) {
            Global_mem[addr] = Address();
        } 

        return Global_mem[addr];
    }
};