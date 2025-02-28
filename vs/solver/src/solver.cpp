#define _CRT_NONSTDC_NO_WARNINGS
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <bits/stdc++.h>
#include <random>
#include <unordered_set>
#include <array>
#include <optional>
#ifdef _MSC_VER
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <conio.h>
#include <ppl.h>
#include <omp.h>
#include <filesystem>
#include <intrin.h>
/* g++ functions */
int __builtin_clz(unsigned int n) { unsigned long index; _BitScanReverse(&index, n); return 31 - index; }
int __builtin_ctz(unsigned int n) { unsigned long index; _BitScanForward(&index, n); return index; }
namespace std { inline int __lg(int __n) { return sizeof(int) * 8 - 1 - __builtin_clz(__n); } }
int __builtin_popcount(int bits) {
    bits = (bits & 0x55555555) + (bits >> 1 & 0x55555555);
    bits = (bits & 0x33333333) + (bits >> 2 & 0x33333333);
    bits = (bits & 0x0f0f0f0f) + (bits >> 4 & 0x0f0f0f0f);
    bits = (bits & 0x00ff00ff) + (bits >> 8 & 0x00ff00ff);
    return (bits & 0x0000ffff) + (bits >> 16 & 0x0000ffff);
}
/* enable __uint128_t in MSVC */
//#include <boost/multiprecision/cpp_int.hpp>
//using __uint128_t = boost::multiprecision::uint128_t;
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro io **/
namespace aux {
    template<typename T, unsigned N, unsigned L> struct tp { static void output(std::ostream& os, const T& v) { os << std::get<N>(v) << ", "; tp<T, N + 1, L>::output(os, v); } };
    template<typename T, unsigned N> struct tp<T, N, N> { static void output(std::ostream& os, const T& v) { os << std::get<N>(v); } };
}
template<typename... Ts> std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) { os << '{'; aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t); return os << '}'; } // tuple out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x); // container out (fwd decl)
template<class S, class T> std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) { return os << '{' << p.first << ", " << p.second << '}'; } // pair out
template<class S, class T> std::istream& operator>>(std::istream& is, std::pair<S, T>& p) { return is >> p.first >> p.second; } // pair in
std::ostream& operator<<(std::ostream& os, const std::vector<bool>::reference& v) { os << (v ? '1' : '0'); return os; } // bool (vector) out
std::ostream& operator<<(std::ostream& os, const std::vector<bool>& v) { bool f = true; os << '{'; for (const auto& x : v) { os << (f ? "" : ", ") << x; f = false; } os << '}'; return os; } // vector<bool> out
template<class Ch, class Tr, class Container> std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) { bool f = true; os << '{'; for (auto& y : x) { os << (f ? "" : ", ") << y; f = false; } return os << '}'; } // container out
template<class T, class = decltype(std::begin(std::declval<T&>())), class = typename std::enable_if<!std::is_same<T, std::string>::value>::type> std::istream& operator>>(std::istream& is, T& a) { for (auto& x : a) is >> x; return is; } // container in
template<typename T> auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) { out << t.stringify(); return out; } // struct (has stringify() func) out
/** io setup **/
struct IOSetup { IOSetup(bool f) { if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); } std::cout << std::fixed << std::setprecision(15); } }
iosetup(true); // set false when solving interective problems
/** string formatter **/
template<typename... Ts> std::string format(const std::string& f, Ts... t) { size_t l = std::snprintf(nullptr, 0, f.c_str(), t...); std::vector<char> b(l + 1); std::snprintf(&b[0], l + 1, f.c_str(), t...); return std::string(&b[0], &b[0] + l); }
/** dump **/
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<']'<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
/** timer **/
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 2.9e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 2.9e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() const { return (time() - t - paused) * 1000.0; }
};
/** rand **/
struct Xorshift {
    Xorshift() {}
    Xorshift(uint64_t seed) { reseed(seed); }
    inline void reseed(uint64_t seed) { x = 0x498b3bc5 ^ seed; for (int i = 0; i < 20; i++) next_u64(); }
    inline uint64_t next_u64() { x ^= x << 7; return x ^= x >> 9; }
    inline uint32_t next_u32() { return next_u64() >> 32; }
    inline uint32_t next_u32(uint32_t mod) { return ((uint64_t)next_u32() * mod) >> 32; }
    inline uint32_t next_u32(uint32_t l, uint32_t r) { return l + next_u32(r - l + 1); }
    inline double next_double() { return next_u32() * e; }
    inline double next_double(double c) { return next_double() * c; }
    inline double next_double(double l, double r) { return next_double(r - l) + l; }
private:
    static constexpr uint32_t M = UINT_MAX;
    static constexpr double e = 1.0 / M;
    uint64_t x = 88172645463325252LL;
};
/** shuffle **/
template<typename T> void shuffle_vector(std::vector<T>& v, Xorshift& rnd) { int n = v.size(); for (int i = n - 1; i >= 1; i--) { int r = rnd.next_u32(i); std::swap(v[i], v[r]); } }
/** split **/
std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> res;
    std::string buf;
    for (const auto& c : str) {
        if (delim.find(c) != std::string::npos) {
            if (!buf.empty()) res.push_back(buf);
            buf.clear();
        }
        else buf += c;
    }
    if (!buf.empty()) res.push_back(buf);
    return res;
}
std::string join(const std::string& delim, const std::vector<std::string>& elems) {
    if (elems.empty()) return "";
    std::string res = elems[0];
    for (int i = 1; i < (int)elems.size(); i++) {
        res += delim + elems[i];
    }
    return res;
}
/** misc **/
template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) { std::fill((T*)array, (T*)(array + N), val); } // fill array
template<typename T, typename ...Args> auto make_vector(T x, int arg, Args ...args) { if constexpr (sizeof...(args) == 0)return std::vector<T>(arg, x); else return std::vector(arg, make_vector<T>(x, args...)); }
template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }
/** hash **/
namespace aux { template<typename T> inline void hash(std::size_t& s, const T& v) { s ^= std::hash<T>()(v) + 0x9e3779b9 + (s << 6) + (s >> 2); } }
namespace std { template<typename F, typename S> struct hash<std::pair<F, S>> { size_t operator()(const std::pair<F, S>& s) const noexcept { size_t seed = 0; aux::hash(seed, s.first); aux::hash(seed, s.second); return seed; } }; }

/* fast queue */
class FastQueue {
    int front = 0;
    int back = 0;
    int v[4096];
public:
    inline bool empty() { return front == back; }
    inline void push(int x) { v[front++] = x; }
    inline int pop() { return v[back++]; }
    inline void reset() { front = back = 0; }
    inline int size() { return front - back; }
};

class RandomQueue {
    int sz = 0;
    int v[4096];
public:
    inline bool empty() const { return !sz; }
    inline int size() const { return sz; }
    inline void push(int x) { v[sz++] = x; }
    inline void reset() { sz = 0; }
    inline int pop(int i) {
        std::swap(v[i], v[sz - 1]);
        return v[--sz];
    }
    inline int pop(Xorshift& rnd) {
        return pop(rnd.next_u32(sz));
    }
};

#if 1
inline double get_temp(double stemp, double etemp, double t, double T) {
    return etemp + (stemp - etemp) * (T - t) / T;
};
#else
inline double get_temp(double stemp, double etemp, double t, double T) {
    return stemp * pow(etemp / stemp, t / T);
};
#endif

struct LogTable {
    static constexpr int M = 65536;
    static constexpr int mask = M - 1;
    double l[M];
    LogTable() : l() {
        unsigned long long x = 88172645463325252ULL;
        double log_u64max = log(2) * 64;
        for (int i = 0; i < M; i++) {
            x = x ^ (x << 7);
            x = x ^ (x >> 9);
            l[i] = log(double(x)) - log_u64max;
        }
    }
    inline double operator[](int i) const { return l[i & mask]; }
} log_table;


struct Point {
    int x, y;
    Point(int x_ = 0, int y_ = 0) : x(x_), y(y_) {}
    inline int dist2(const Point& rhs) const {
        return (x - rhs.x) * (x - rhs.x) + (y - rhs.y) * (y - rhs.y);
    }
};

Point nearest_lattice_point(const Point& from, const Point& to, const int radius) {
    const auto& [xfrom, yfrom] = from;
    const auto& [xto, yto] = to;
    int mindist2 = INT_MAX;
    int minx = -1, miny = -1;
    for (int x = xfrom - radius; x <= xfrom + radius; x++) {
        for (int y = yfrom - radius; y <= yfrom + radius; y++) {
            if ((x - xfrom) * (x - xfrom) + (y - yfrom) * (y - yfrom) > radius * radius) continue;
            int dist2 = (x - xto) * (x - xto) + (y - yto) * (y - yto);
            if (chmin(mindist2, dist2)) {
                minx = x;
                miny = y;
            }
        }
    }
    return { minx, miny };
}

Point near_lattice_point(const Point& from, const Point& to, const int radius) {
    const auto& [xfrom, yfrom] = from;
    const auto& [xto, yto] = to;
    if ((xfrom - xto) * (xfrom - xto) + (yfrom - yto) * (yfrom - yto) <= radius * radius) {
        return { xto, yto };
    }
    double dx = xto - xfrom, dy = yto - yfrom, d = sqrt(dx * dx + dy * dy);
    int cx = (int)round(xfrom + dx * radius / d), cy = (int)round(yfrom + dy * radius / d);
    int mindist2 = INT_MAX;
    int minx = -1, miny = -1;
    for (int x = cx - 1; x <= cx + 1; x++) {
        for (int y = cy - 1; y <= cy + 1; y++) {
            if ((x - xfrom) * (x - xfrom) + (y - yfrom) * (y - yfrom) > radius * radius) continue;
            int dist2 = (x - xto) * (x - xto) + (y - yto) * (y - yto);
            if (chmin(mindist2, dist2)) {
                minx = x;
                miny = y;
            }
        }
    }
    return { minx, miny };
}

void test_nearest_lattice_point() {
    int all = 0, wrong = 0;
    int xfrom = 0, yfrom = 0;
    for (int radius = 1; radius <= 30; radius++) {
        for (int dx = 0; dx <= 100; dx++) {
            for (int dy = 0; dy <= 100; dy++) {
                int xto = xfrom + dx, yto = yfrom + dy;
                auto [x1, y1] = nearest_lattice_point({ xfrom, yfrom }, { xto, yto }, radius);
                auto [x2, y2] = near_lattice_point({ xfrom, yfrom }, { xto, yto }, radius);
                if (x1 != x2 || y1 != y2) {
                    //dump(xfrom, yfrom, xto, yto, radius, dx, dy, x1, y1, x2, y2);
                    wrong++;
                }
                all++;
                //assert(x1 == x2 && y1 == y2);
            }
        }
    }
    dump(wrong, all);
}

// The world width and height, the number of monsters, and the number of turns are each less than 3000.
// The hero's coordinates should always be integers during this glorious journey.
struct HeroInfo {

    const int base_speed;
    const int base_power;
    const int base_range;
    const int level_speed_coeff;
    const int level_power_coeff;
    const int level_range_coeff;

private:
    HeroInfo(const int base_speed_, const int base_power_, const int base_range_, const int level_speed_coeff_, const int level_power_coeff_, const int level_range_coeff_)
        : base_speed(base_speed_), base_power(base_power_), base_range(base_range_), level_speed_coeff(level_speed_coeff_), level_power_coeff(level_power_coeff_), level_range_coeff(level_range_coeff_) {}

public:
    static HeroInfo load(nlohmann::json j) {
        const int base_speed = j["base_speed"];
        const int base_power = j["base_power"];
        const int base_range = j["base_range"];
        const int level_speed_coeff = j["level_speed_coeff"];
        const int level_power_coeff = j["level_power_coeff"];
        const int level_range_coeff = j["level_range_coeff"];
        return HeroInfo(base_speed, base_power, base_range, level_speed_coeff, level_power_coeff, level_range_coeff);
    }

};

struct MonsterInfo {

    const Point pos;
    const int hp;
    const int gold;
    const int xp;

private:
    MonsterInfo(const Point pos_, const int hp_, const int gold_, const int exp_)
        : pos(pos_), hp(hp_), gold(gold_), xp(exp_) {}

public:
    static MonsterInfo load(nlohmann::json j) {
        const int x = j["x"];
        const int y = j["y"];
        const int hp = j["hp"];
        const int gold = j["gold"];
        const int exp = j["exp"];
        return MonsterInfo(Point(x, y), hp, gold, exp);
    }

};

struct Input {

    const int num_turns;
    const int width;
    const int height;
    const Point start_pos;
    HeroInfo hero;
    std::vector<MonsterInfo> monsters;

private:
    Input(const int num_turns_, const int width_, const int height_, const Point start_pos_, const HeroInfo& hero_, const std::vector<MonsterInfo>& monsters_)
        : num_turns(num_turns_), width(width_), height(height_), start_pos(start_pos_), hero(hero_), monsters(monsters_) {}

public:
    static Input load(nlohmann::json j) {
        const int num_turns = j["num_turns"];
        const int width = j["width"];
        const int height = j["height"];
        const int start_x = j["start_x"];
        const int start_y = j["start_y"];
        const auto hero = HeroInfo::load(j["hero"]);
        std::vector<MonsterInfo> monsters;
        for (nlohmann::json jj : j["monsters"]) {
            auto monster = MonsterInfo::load(jj);
            monsters.push_back(monster);
        }
        return Input(num_turns, width, height, Point(start_x, start_y), hero, monsters);
    }

};


struct Hero {

    const HeroInfo& info;
    int level;
    int xp;
    int gold;

    Hero(const HeroInfo& info_) : info(info_), level(0), xp(0), gold(0) {}

    inline int speed() const {
        return info.base_speed + info.base_speed * level * info.level_speed_coeff / 100;
    }

    inline int power() const {
        return info.base_power + info.base_power * level * info.level_power_coeff / 100;
    }

    inline int range() const {
        return info.base_range + info.base_range * level * info.level_range_coeff / 100;
    }

    void update() {
        while (true) {
            const int required = 1000 + level * (level + 1) * 50;
            if (required <= xp) {
                xp -= required;
                level++;
            }
            else {
                break;
            }
        }
    }

};

struct Monster {

    const MonsterInfo& info;
    int hp;
    
    Monster(const MonsterInfo& info_) : info(info_), hp(info.hp) {}

};


struct Action {

    enum struct Type { MOVE, ATTACK };

    Type type;
    int x, y, id;

    static Action move(int x_, int y_) {
        return { Type::MOVE, x_, y_, -1 };
    }

    static Action attack(int id_) {
        return { Type::ATTACK, -1, -1, id_ };
    }

    static Action load(nlohmann::json j) {
        Action action;
        if (j["type"] == "move") {
            action.type = Action::Type::MOVE;
            action.x = j["target_x"];
            action.y = j["target_y"];
            action.id = -1;
        }
        else if (j["type"] == "attack") {
            action.type = Action::Type::ATTACK;
            action.x = -1;
            action.y = -1;
            action.id = j["target_id"];
        }
        else {
            assert(false);
        }
        return action;
    }

    nlohmann::json to_json() const {
        nlohmann::json j;
        if (type == Type::MOVE) {
            j["type"] = "move";
            j["target_x"] = x;
            j["target_y"] = y;
        }
        else {
            j["type"] = "attack";
            j["target_id"] = id;
        }
        return j;
    }

};

struct Output {

    std::vector<Action> actions;

    static Output load(nlohmann::json j) {
        std::vector<Action> actions;
        for (nlohmann::json jj : j["moves"]) {
            actions.push_back(Action::load(jj));
        }
        return { actions };
    }

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["moves"] = {};
        for (const auto& action : actions) {
            j["moves"].push_back(action.to_json());
        }
        return j;
    }

};

void output_solution(const Output& output, const std::string output_filename) {
    std::ofstream output_file(output_filename);
    nlohmann::json output_json = output.to_json();
    output_file << output_json;
}

Input load_input(int seed) {
    std::string input_filename(format("../../in/%03d.json", seed));
    std::ifstream input_file(input_filename);
    nlohmann::json input_json;
    input_file >> input_json;
    return Input::load(input_json);
}

std::optional<Output> load_output(int seed) {
    std::string output_filename(format("../../out/%03d.json", seed));
    if (std::filesystem::exists(output_filename)) {
        std::ifstream output_file(output_filename);
        nlohmann::json output_json;
        output_file >> output_json;
        return Output::load(output_json);
    }
    return std::nullopt;
}

int compute_score(const Input& input, const Output& output) {
    Hero hero(input.hero);
    auto [x, y] = input.start_pos;
    std::vector<Monster> monsters;
    for (const auto& minfo : input.monsters) {
        monsters.emplace_back(minfo);
    }
    assert(output.actions.size() <= input.num_turns); // should be equal?
    for (const auto& action : output.actions) {
        if (action.type == Action::Type::MOVE) {
            const int nx = action.x, ny = action.y;
            assert(0 <= nx && nx <= input.width && 0 <= ny && ny <= input.height);
            const int speed = hero.speed();
            assert((x - nx) * (x - nx) + (y - ny) * (y - ny) <= speed * speed);
            x = nx;
            y = ny;
        }
        else {
            int id = action.id;
            assert(0 <= id && id < (int)monsters.size());
            auto& monster = monsters[id];
            assert(monster.hp > 0);
            const auto [mx, my] = monster.info.pos;
            const int range = hero.range();
            assert((x - mx) * (x - mx) + (y - my) * (y - my) <= range * range);
            monster.hp -= hero.power();
            if (monster.hp <= 0) { // dead
                hero.xp += monster.info.xp;
                hero.gold += monster.info.gold;
                hero.update();
            }
        }
    }
    return hero.gold;
}

int get_best_score(int seed) {
    int best_score = -1;
    auto input = load_input(seed);
    auto output_best_opt = load_output(seed);
    if (output_best_opt) {
        best_score = compute_score(input, output_best_opt.value());
    }
    return best_score;
}

void output_if_best(int seed, const Output& output) {
    int best_score = get_best_score(seed);
    auto input = load_input(seed);
    int score = compute_score(input, output);
    if (best_score < score) {
        std::cerr << format("seed %3d: %9d -> %9d (+%2.4f%%)\n", seed, best_score, score, (score - best_score) * 100.0 / best_score);
        output_solution(output, format("../../out/%03d.json", seed));
    }
}

void test_sample() {

    std::string input_filename("../../in/000.json");
    std::ifstream input_file(input_filename);
    nlohmann::json input_json;
    input_file >> input_json;

    auto input = Input::load(input_json);

    std::string output_filename("../../out/000.json");
    std::ifstream output_file(output_filename);
    nlohmann::json output_json;
    output_file >> output_json;

    auto output = Output::load(output_json);

    assert(compute_score(input, output) == 664);

}

struct State {

    const Input& input;
    Hero hero;
    Point pos;
    std::vector<Monster> monsters;
    std::vector<Action> actions;
    int target_mid = -1;
    int damage = 0;

    State(const Input& input_) : input(input_), hero(input.hero), pos(input.start_pos) {
        for (const auto& minfo : input.monsters) {
            monsters.emplace_back(minfo);
        }
    }

    int get_nearest_monster_id() const {
        int mindist2 = INT_MAX;
        int id = -1;
        for (int mid = 0; mid < (int)monsters.size(); mid++) {
            const auto& monster = monsters[mid];
            if (monster.hp <= 0) continue;
            if (chmin(mindist2, pos.dist2(monster.info.pos))) {
                id = mid;
            }
        }
        return id;
    }

    bool all_monsters_are_dead() const {
        for (const auto& monster : monsters) {
            if (monster.hp > 0) return false;
        }
        return true;
    }

    bool can_attack(int mid) const {
        const Monster& monster = monsters[mid];
        assert(monster.hp > 0);
        const int range = hero.range();
        return pos.dist2(monster.info.pos) <= range * range;
    }

    void attack(int mid) {
        Monster& monster = monsters[mid];
        int power = hero.power();
        monster.hp -= power;
        target_mid = mid;
        damage += power;
        if (monster.hp <= 0) {
            damage = 0;
            hero.xp += monster.info.xp;
            hero.gold += monster.info.gold;
            hero.update();
        }
        actions.push_back(Action::attack(mid));
    }

    void move_to_monster(int mid) {
        const Monster& monster = monsters[mid];
        auto& [x, y] = pos;
        const auto [mx, my] = monster.info.pos;
        const int speed = hero.speed();
        int mindist2 = INT_MAX;
        int tx = -1, ty = -1;
        for (int nx = std::max(0, x - speed); nx <= std::min(input.width, x + speed); nx++) {
            for (int ny = std::max(0, y - speed); ny <= std::min(input.height, y + speed); ny++) {
                if ((x - nx) * (x - nx) + (y - ny) * (y - ny) > speed * speed) continue;
                const int dist2 = (nx - mx) * (nx - mx) + (ny - my) * (ny - my);
                if (chmin(mindist2, dist2)) {
                    tx = nx;
                    ty = ny;
                }
            }
        }
        x = tx;
        y = ty;
        actions.push_back(Action::move(x, y));
    }

    void move_to_monster2(int mid) {
        const Monster& monster = monsters[mid];
        const int speed = hero.speed();
        pos = near_lattice_point(pos, monster.info.pos, speed);
        actions.push_back(Action::move(pos.x, pos.y));
    }

    void solve_nearest_neighbor(int first_monster_id = -1) {
        bool reached = false;
        if (first_monster_id == -1) {
            reached = true;
        }
        for (int turn = 0; turn < input.num_turns; turn++) {
            if (!reached) {
                if (!can_attack(first_monster_id)) {
                    move_to_monster2(first_monster_id);
                    continue;
                }
                else {
                    reached = true;
                }
            }
            if (all_monsters_are_dead()) {
                actions.push_back(Action::move(pos.x, pos.y)); // do nothing
            }
            else {
                int mid = get_nearest_monster_id();
                if (can_attack(mid)) {
                    attack(mid);
                }
                else {
                    //move_to_monster(mid);
                    move_to_monster2(mid);
                }
            }
            //dump(turn, hero.level, hero.gold, actions.back().to_json());
        }
    }

    void solve_with_order(const std::vector<int>& order) {
        int turn = 0;
        for (int mid : order) {
            while (monsters[mid].hp > 0) {
                if (can_attack(mid)) {
                    attack(mid);
                }
                else {
                    move_to_monster2(mid);
                }
                turn++;
                if (input.num_turns <= turn) return;
            }
        }
        while (turn < input.num_turns) {
            actions.push_back(Action::move(pos.x, pos.y));
        }
    }

    double eval() const {
        double score = hero.gold;
        if (damage) {
            const auto& monster = monsters[target_mid];
            score += 1.0 * monster.info.gold * damage / monster.hp;
        }
        return score;
    }

};

Output solve(const Input& input, const int seed = -1, const double duration_ms = 100000, const double temp_ratio = 0.01) {

    Timer timer;

    double best_score = -1.0;
    std::vector<Action> best_actions;

    //for (int mid = -1; mid < (int)input.monsters.size(); mid++) {
    //    State state(input);
    //    state.solve_nearest_neighbor(mid);
    //    int score = compute_score(input, { state.actions });
    //    if (chmax(best_score, score)) {
    //        best_actions = state.actions;
    //        //dump(mid, best_score);
    //    }
    //}
    //return { best_actions };

    std::vector<int> order(input.monsters.size());
    std::iota(order.begin(), order.end(), 0);

    {
        if (true) {
            //State state(input);
            //state.solve_nearest_neighbor();
            //best_score = state.hero.gold;
            //best_actions = state.actions;
            if (seed != -1) {
                auto output_best_opt = load_output(seed);
                if (output_best_opt) {
                    int score = compute_score(input, output_best_opt.value());
                    if (chmax(best_score, (double)score)) {
                        best_actions = output_best_opt.value().actions;
                    }
                }
            }
        }
        //dump(best_score);
        order.clear();
        std::set<int> seen;
        for (const auto& action : best_actions) {
            if (action.type == Action::Type::ATTACK) {
                if (!seen.count(action.id)) {
                    seen.insert(action.id);
                    order.push_back(action.id);
                }
            }
        }
        for (int mid = 0; mid < input.monsters.size(); mid++) {
            if (!seen.count(mid)) {
                seen.insert(mid);
                order.push_back(mid);
            }
        }
    }

    double prev_score = best_score;

    Xorshift rnd;
    int loop = 0;
    const int num_monsters = input.monsters.size();
    double start_time = timer.elapsed_ms(), now_time, end_time = start_time + duration_ms;
    double start_temp = best_score * temp_ratio;
    double end_temp = start_temp * 0.1;
    //dump(start_temp);
    while ((now_time = timer.elapsed_ms()) < end_time) {
        loop++;

        int i, j;
        do {
            i = rnd.next_u32(0, num_monsters - 1);
            j = rnd.next_u32(1, num_monsters);
        } while (i == j);
        if (i > j) std::swap(i, j);

        if (!rnd.next_u32(10)) {
            std::reverse(order.begin() + i, order.begin() + j);
            State state(input);
            state.solve_with_order(order);
            int diff = state.hero.gold - prev_score;
            double temp = get_temp(start_temp, end_temp, now_time - start_time, end_time - start_time);
            double prob = exp(diff / temp);
            if (rnd.next_double() < prob) {
                prev_score = state.hero.gold;
                if (chmax(best_score, state.eval())) {
                    best_actions = state.actions;
                    //dump("rev", timer.elapsed_ms(), loop, best_score);
                }
            }
            else {
                std::reverse(order.begin() + i, order.begin() + j);
            }
        }
        else if (j < num_monsters) {
            std::swap(order[i], order[j]);
            State state(input);
            state.solve_with_order(order);
            int diff = state.hero.gold - prev_score;
            double temp = get_temp(start_temp, end_temp, now_time - start_time, end_time - start_time);
            double prob = exp(diff / temp);
            if (rnd.next_double() < prob) {
                prev_score = state.hero.gold;
                if (chmax(best_score, state.eval())) {
                    best_actions = state.actions;
                    //dump("swp", timer.elapsed_ms(), loop, best_score);
                }
            }
            else {
                std::swap(order[i], order[j]);
            }
        }

        if (!(loop & 0xFFF)) {
            //dump(timer.elapsed_ms(), loop, prev_score);
        }

    }

    return { best_actions };
}

void batch_execute() {
    const int batch_size = 5;
    const int num_seeds = 25;
    concurrency::critical_section mtx;
    for (int begin = 1; begin <= num_seeds; begin += batch_size) {
        int end = std::min(begin + batch_size, num_seeds + 1);
        dump(begin, end);
        concurrency::parallel_for(begin, end, [&](int seed) {
            auto input = load_input(seed);
            auto output = solve(input, seed, 60000, 0.0002);
            {
                mtx.lock();
                output_if_best(seed, output);
                mtx.unlock();
            }
        });
    }
}

void batch_execute2() {
    const int batch_size = 5;
    std::vector<int> seeds/*, best_scores, next_seeds, next_best_scores*/;
    for (int seed = 1; seed <= 25; seed++) {
        seeds.push_back(seed);
        //best_scores.push_back(get_best_score(seed));
    }
    int iter = 0;
    while (!seeds.empty()) {
        //next_seeds.clear();
        //next_best_scores.clear();
        dump(iter);
        //dump(best_scores);
        concurrency::critical_section mtx;
        for (int begin = 0; begin < (int)seeds.size(); begin += batch_size) {
            int end = std::min(begin + batch_size, (int)seeds.size());
            dump(begin, end);
            concurrency::parallel_for(begin, end, [&seeds, /*&best_scores, &next_seeds, &next_best_scores,*/ &mtx](int id) {
                int seed = seeds[id];
                //int best_score = best_scores[id];
                auto input = load_input(seed);
                auto output = solve(input, seed, 60000, 0.0001);
                int score = compute_score(input, output);
                {
                    mtx.lock();
                    output_if_best(seed, output);
                    //if (best_score < score) {
                    //    next_seeds.push_back(seed);
                    //    next_best_scores.push_back(score);
                    //}
                    mtx.unlock();
                }
                });
        }
        //seeds = next_seeds;
        //best_scores = next_best_scores;
        iter++;
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

#if 1
    batch_execute2();
    exit(0);
#endif

    Timer timer;

#ifdef HAVE_OPENCV_HIGHGUI
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

    const int seed = 14;
    auto input = load_input(seed);
    auto output = solve(input, seed, 60000, 0.00001);

    output_if_best(seed, output);

    return 0;
}