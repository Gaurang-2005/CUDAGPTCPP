#pragma once

#include "tensor/tensor.hpp"
#include <unordered_map>

class Node {
public:
    tensorMem* val;
    Node* next = nullptr;
    Node* prev = nullptr;
    Node(tensorMem* val) : val(val) {}
};

class lru {
    Node* head = nullptr;
    Node* back = nullptr;
    std::unordered_map<tensorMem*, Node*> present;
public:
    void push(tensorMem* val) {
        if (!val) return;
        if (present.find(val) != present.end()) {
            if (back == present[val]) return;
            if (head == present[val]) {
                back -> next = head;
                back -> next -> prev = back;
                back = back -> next;
                head = head -> next;
                head -> prev = nullptr;
                back -> next = nullptr;
            }
            else {
                auto temp1 = present[val] -> prev;
                present[val] -> prev = back;
                temp1 -> next = present[val] -> next;
                present[val] -> next -> prev = temp1;
                present[val] -> next = nullptr;
                back -> next = present[val];
                back = present[val];
            }
            return;
        }
        if (!head) {
            head = new Node(val);
            present[val] = head;
            back = head;
        }
        else {
            back -> next = new Node(val);
            present[val] = back -> next;
            auto temp = back;
            back = back -> next;
            back -> prev = temp;
        }
    }
    void pop() {
        if (!head) return;
        if (!head -> next) {
            present.erase(head -> val);
            delete head;
            head = nullptr;
            back = nullptr;
            return;
        }
        auto temp = head -> next;
        present.erase(head -> val);
        delete head;
        head = temp;
        head -> prev = nullptr;
    }
    void remove(tensorMem* val) {
        if (present.find(val) == present.end()) return;
        if (present[val] == back) {
            if (head == back) {
                pop();
                return;
            }
            auto temp = present[val] -> prev;
            temp -> next = nullptr;
            delete present[val];
            present.erase(val);
            back = temp;
        }
        else if (present[val] -> prev) {
            auto temp = present[val] -> prev;
            temp -> next = present[val] -> next;
            temp -> next -> prev = temp;
            delete present[val];
            present.erase(val);
        }
        else {
            pop();
        }
    }
    tensorMem* top() {
        if (!head) return nullptr;
        return head->val;
    }
};
#include <thread>
class memoryManager {
    std::thread worker;
    lru qu;
    bool memManagerActive = false;
    size_t threshold;
    memoryManager() {
        worker = std::thread(&memoryManager::evict, this);
        threshold = 1073741824;
    }
public:
    static memoryManager& get() {
        static memoryManager instance;
        return instance;
    }
    void registerTensor(tensorMem* ptr) {
        qu.push(ptr);
    }
    void unregisterTensor(tensorMem* ptr) {
        qu.remove(ptr);
    }
    void setThreshold(size_t bytes) {
        threshold = bytes;
    }
    void evict();
};