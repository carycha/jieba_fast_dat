#ifndef JIEBA_FAST_DAT_UTILS_HPP
#define JIEBA_FAST_DAT_UTILS_HPP

#include <string>
#include <vector>
#include <codecvt>
#include <locale>
#include <cstdint>

namespace jieba_fast_dat {

// Helper to get byte offsets for each character in a UTF-8 string
inline std::vector<size_t> get_utf8_offsets(const std::string& s) {
    std::vector<size_t> offsets;
    offsets.reserve(s.size() + 1);
    for (size_t i = 0; i < s.size(); ) {
        offsets.push_back(i);
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) i += 1;
        else if ((c & 0xE0) == 0xC0) i += 2;
        else if ((c & 0xF0) == 0xE0) i += 3;
        else if ((c & 0xF8) == 0xF0) i += 4;
        else i += 1; // Should not happen in valid UTF-8
    }
    offsets.push_back(s.size());
    return offsets;
}

// Helper to get next char32_t from UTF-8 string
inline char32_t utf8_next_char(const char*& p, const char* end) {
    if (p >= end) return 0;
    unsigned char c = static_cast<unsigned char>(*p++);
    if (c < 0x80) return c;
    if ((c & 0xE0) == 0xC0) {
        if (p >= end) return c;
        char32_t res = (c & 0x1F) << 6;
        res |= (*p++ & 0x3F);
        return res;
    }
    if ((c & 0xF0) == 0xE0) {
        if (p + 1 >= end) return c;
        char32_t res = (c & 0x0F) << 12;
        res |= (*p++ & 0x3F) << 6;
        res |= (*p++ & 0x3F);
        return res;
    }
    if ((c & 0xF8) == 0xF0) {
        if (p + 2 >= end) return c;
        char32_t res = (c & 0x07) << 18;
        res |= (*p++ & 0x3F) << 12;
        res |= (*p++ & 0x3F) << 6;
        res |= (*p++ & 0x3F);
        return res;
    }
    return c;
}

// Helper to convert u32string to UTF-8 string
inline std::string u32_to_utf8(const std::u32string& s) {
    std::string res;
    for (char32_t ch : s) {
        if (ch < 0x80) res += static_cast<char>(ch);
        else if (ch < 0x800) {
            res += static_cast<char>(0xC0 | (ch >> 6));
            res += static_cast<char>(0x80 | (ch & 0x3F));
        } else if (ch < 0x10000) {
            res += static_cast<char>(0xE0 | (ch >> 12));
            res += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
            res += static_cast<char>(0x80 | (ch & 0x3F));
        } else {
            res += static_cast<char>(0xF0 | (ch >> 18));
            res += static_cast<char>(0x80 | ((ch >> 12) & 0x3F));
            res += static_cast<char>(0x80 | ((ch >> 6) & 0x3F));
            res += static_cast<char>(0x80 | (ch & 0x3F));
        }
    }
    return res;
}

// Helper to convert UTF-8 string to u32string
inline std::u32string utf8_to_u32(const std::string& s) {
    std::u32string res;
    const char* p = s.data();
    const char* end = p + s.size();
    while (p < end) {
        char32_t ch = utf8_next_char(p, end);
        if (ch != 0 || p < end) res += ch;
    }
    return res;
}

inline bool is_han_alnum_fast(char32_t ch) {
    return (ch >= 0x4E00 && ch <= 0x9FD5) ||
           (ch >= U'a' && ch <= U'z') ||
           (ch >= U'A' && ch <= U'Z') ||
           (ch >= U'0' && ch <= U'9') ||
           ch == U'+' || ch == U'#' || ch == U'&' || ch == U'.' || ch == U'#' || ch == U'_' || ch == U'-' || ch == U'%';
}

inline bool is_eng_fast(char32_t ch) {
    return (ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z') || (ch >= U'0' && ch <= U'9');
}

inline bool is_number(const std::u32string& s) {
    if (s.empty()) return false;
    for (char32_t ch : s) {
        if (!((ch >= U'0' && ch <= U'9') || ch == U'.')) {
            return false;
        }
    }
    return true;
}

inline bool is_english(const std::u32string& s) {
    if (s.empty()) return false;
    bool has_alpha = false;
    for (char32_t ch : s) {
        if ((ch >= U'a' && ch <= U'z') || (ch >= U'A' && ch <= U'Z')) {
            has_alpha = true;
        } else if (ch >= U'0' && ch <= U'9') {
            // digit allowed
        } else {
            return false;
        }
    }
    return has_alpha;
}

} // namespace jieba_fast_dat

#endif // JIEBA_FAST_DAT_UTILS_HPP
