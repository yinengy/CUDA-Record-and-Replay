/* 
 * a data race checker
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */
 
#include <unordered_set>
#include <unordered_map>
#include <stdint.h>

class Checker {
private:
    struct Instruction {
        int func_id;
        int inst_id;

        Instruction() {
            func_id = 0;
            inst_id = 0;
        }

        bool operator==(const Instruction &other) const { 
            return (func_id == other.func_id &&
                    inst_id == other.inst_id);
        }
    }

    class HashInstruction { 
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
            insts = std::unordered_set<Instruction>();
        }
    }

    // SFR in a block
    struct SFR {
        int block_id;
        int SFR_id;

        SFR(int block_id. int SFR_id) {
            this->block_id = block_id;
            this->SFR_id = SFR_id;
        }

        bool operator==(const SFR &other) const { 
            return (block_id == other.block_id &&
                    SFR_id == other.SFR_id);
        }
    }

    class HashSFR { 
        size_t operator()(const SFR &i) const
        { 
            return (std::hash<int>()(i.block_id)) ^  
                (std::hash<int>()(i.SFR_id)); 
        } 
    }; 

    // key: SFR, val: shared_mem (a dic of addr : Address (has two set of Thread ID))
    std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR> SFR_shared_mem;

    // key: SFR, val: global_mem (a dic of addr : Address (has two set of Thread ID))
    std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR> SFR_shared_mem;

    // key: addr, val: Address (has two set of Block ID)
    std::unordered_map<uint64_t, Address> global_mem;

public:
    Checker() {
        SFR_shared_mem = std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR>();
        SFR_shared_mem = std::unordered_map<SFR, std::unordered_map<uint64_t, Address>, HashSFR>();
        GLOBAL_mem = std::unordered_map<uint64_t, Address>();
    }

}