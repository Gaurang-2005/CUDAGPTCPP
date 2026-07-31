    #pragma once

    #include "tensor/tensor.hpp"
    #include <unordered_map>
    #include <functional>
    #include "tokenizer/types.hpp"

    struct pairHash {
        size_t operator()(const TokenPair& p) const noexcept {
            size_t h1 = std::hash<TokenID>{}(p.first);
            size_t h2 = std::hash<TokenID>{}(p.second);

            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    struct MergeInfo {
        TokenID token;
        size_t rank;
    };

    class BPE {
        size_t vocabSize;
        std::unordered_map<TokenID, TokenPair> mergeRules;
        std::unordered_map<TokenPair, MergeInfo, pairHash> pairToMerge;

        void expand(const TokenID& id, std::string& out) const {
            if (id < 256) {
                out.push_back(char(id));
                return;
            }
            auto pair = mergeRules.at(id);
            expand(pair.first, out);
            expand(pair.second, out);
        }

    public:
        BPE(const size_t vocabSize) : vocabSize(vocabSize) {}

        void train(const std::vector<std::string>& corpus) {
            //encoder
            std::vector<std::vector<TokenID>> encodedCorpus;
            for (size_t i = 0; i < corpus.size(); ++i) {
                std::vector<TokenID> tempCorp;
                for (size_t j = 0; j < corpus[i].length(); ++j) {
                    tempCorp.push_back(TokenID((unsigned char)(corpus[i][j])));
                } 
                encodedCorpus.push_back(tempCorp);
            }

            //adjacent pairs
            size_t nextToken = 255;
            size_t rank = 0;
            while (nextToken < vocabSize) {
                std::unordered_map<TokenPair, size_t, pairHash> freq;

                for (const auto& doc : encodedCorpus) {
                    for (size_t j = 0; j + 1 < doc.size(); j++) {
                        freq[TokenPair(doc[j], doc[j + 1])]++;
                    }
                }
                if (freq.empty()) break;
                TokenPair maxPair = freq.begin()->first;
                size_t maxFreq = freq.begin()->second;
                for (auto& [key, val] : freq) {
                    if (maxFreq < val) {
                        maxFreq = val;
                        maxPair = key;
                    }
                }
                if (maxFreq < 2) break;
                nextToken++;
                mergeRules[nextToken] = maxPair;
                pairToMerge[maxPair] = {nextToken, rank++};
                for (auto& doc : encodedCorpus) {
                    std::vector<TokenID> newDoc;
                    size_t i = 0;
                    while (i + 1 < doc.size()) {
                        if (maxPair.first == doc[i] && maxPair.second == doc[i + 1]) {
                            newDoc.push_back(nextToken);
                            i+=2;
                        }
                        else {
                            newDoc.push_back(doc[i]);
                            i++;
                        }
                    }
                    if (i < doc.size()) {
                        newDoc.push_back(doc[i]);
                    }
                    doc = std::move(newDoc);
                }
            }
        }

        std::vector<TokenID> encode(const std::string& input) const {

            assert (pairToMerge.size() > 0);

            std::vector<TokenID> inDoc;
            for (size_t i = 0; i < input.size(); ++i) {
                    inDoc.push_back(TokenID((unsigned char)(input[i])));
            }

            while (true) {
                TokenPair rankPair = pairToMerge.begin() -> first;
                MergeInfo info = pairToMerge.begin() -> second;
                bool mergePres = false;

                for (size_t i = 0; i + 1 < inDoc.size(); i++) {
                    auto it = pairToMerge.find({inDoc[i], inDoc[i + 1]});
                    if (it == pairToMerge.end()) continue;
                    if (it->second.rank <= info.rank) {
                        rankPair = it->first;
                        info = it->second;
                        mergePres = true;
                    }
                }

                if (!mergePres) break;
                std::vector<TokenID> newDoc;
                newDoc.reserve(inDoc.size());
                size_t i = 0;
                while (i + 1 < inDoc.size()) {
                    if (rankPair.first == inDoc[i] && rankPair.second == inDoc[i + 1]) {
                        newDoc.push_back(info.token);
                        i+=2;
                    }
                    else {
                        newDoc.push_back(inDoc[i]);
                        i++;
                    }
                }
                if (i < inDoc.size()) {
                    newDoc.push_back(inDoc[i]);
                }
                inDoc = std::move(newDoc);
            }

            return inDoc;

        }

        std::string decode(const std::vector<TokenID>& input) const {
            std::string out;

            for (auto& i : input) {
                expand(i, out);
            }

            return out;
        }

        void save(const std::string& filename) const {
            std::ofstream out(filename, std::ios::binary);
            if (!out)
                throw std::runtime_error("Failed to open file for writing.");

            out.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));

            uint64_t mergeCount = mergeRules.size();
            out.write(reinterpret_cast<const char*>(&mergeCount), sizeof(mergeCount));

            for (TokenID token = 256; token < 256 + mergeCount; ++token) {
                auto it = mergeRules.find(token);
                if (it == mergeRules.end())
                    continue;

                out.write(reinterpret_cast<const char*>(&token), sizeof(TokenID));
                out.write(reinterpret_cast<const char*>(&it->second.first), sizeof(TokenID));
                out.write(reinterpret_cast<const char*>(&it->second.second), sizeof(TokenID));
            }          
        }
        void load(const std::string& filename) {
            std::ifstream in(filename, std::ios::binary);
            if (!in)
                throw std::runtime_error("Failed to open file for reading.");
                
            mergeRules.clear();
            pairToMerge.clear();

            in.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));

            uint64_t mergeCount;
            in.read(reinterpret_cast<char*>(&mergeCount), sizeof(mergeCount));

            size_t rank = 0;

            for (uint64_t i = 0; i < mergeCount; ++i)
            {
                TokenID token;
                TokenID left;
                TokenID right;

                in.read(reinterpret_cast<char*>(&token), sizeof(TokenID));
                in.read(reinterpret_cast<char*>(&left), sizeof(TokenID));
                in.read(reinterpret_cast<char*>(&right), sizeof(TokenID));

                mergeRules[token] = {left, right};
                pairToMerge[{left, right}] = {token, rank++};
            }
        }      
    };

