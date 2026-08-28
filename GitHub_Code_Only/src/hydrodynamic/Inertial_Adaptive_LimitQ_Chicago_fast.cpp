// newRain_fast.cpp  —— 初始干燥 + 降雨三模式/外部雨型 + 过程输出 + 保守并行
// Fast-safe variant: keep the numerical update order unchanged, but throttle
// adaptive progress logging to avoid millions of flushed log lines.
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// ---------------- 配置宏 ----------------
#define LIMIT_Q 0
#define LIMITER 1

// -------------- 工具函数 --------------
template <class Type>
Type StringToNum(const string& str) {
    istringstream iss(str);
    Type num; iss >> num; return num;
}
template<typename T> string toString(const T& t) { ostringstream oss; oss<<t; return oss.str(); }

// -------------- 数据结构 --------------
struct Hydro_cell {
    double Storage=0, Dem=0, Runout=0, Runin=0, Absolute_h=0;
    double q_right_last=0, q_left_last=0, q_up_last=0, q_botton_last=0;
    double q_right_now=0,  q_left_now=0,  q_up_now=0,  q_botton_now=0;
    double emax_left=0, emax_right=0, emax_up=0, emax_down=0;
    double dh=0;
    double wet_elapsed=0; // Horton 机会时间(s)：仅在该格点有可供入渗水时累积
    int    landcover=0;
    float  n=0.012f;
    int    noData=0;
};

struct OPTION {
    string time_adaptive="OFF";
    double res=1.0;
};

// -------------- 降雨调度 --------------
enum class RainMode { UNIFORM, FRONT, BACK, FILE };
struct RainPulse { double t0, t1, frac; };
struct RainInterval { double t0, t1, rate; };

struct RainScheduler {
    RainMode mode = RainMode::UNIFORM;
    double H=0.0;   // 总雨深(m)
    double T=0.0;   // 历时(s)
    double r_uniform=0.0;
    vector<RainPulse> pulses;
    vector<RainInterval> intervals;

    void init(RainMode m, double Hm, double Tm) {
        mode=m; H=Hm; T=Tm; r_uniform = (T>0? H/T : 0.0);
        pulses.clear();
        intervals.clear();
        if (mode==RainMode::FRONT){
            pulses = { {0, 10*60.0, 0.60}, {10*60.0, 30*60.0, 0.30}, {30*60.0, 50*60.0, 0.10}, {50*60.0, 60*60.0, 0.0} };
        } else if (mode==RainMode::BACK){
            pulses = { {0, 10*60.0, 0.00}, {10*60.0, 30*60.0, 0.10}, {30*60.0, 50*60.0, 0.30}, {50*60.0, 60*60.0, 0.60} };
        }
        // 归一化
        if (!pulses.empty()){
            double s=0; for(auto &p: pulses) s+=p.frac;
            if (s>0) for(auto &p: pulses) p.frac/=s;
        }
    }

    bool load_file(const string& path) {
        ifstream in(path.c_str());
        if (!in) {
            cerr<<"Cannot open rain file: "<<path<<"\n";
            return false;
        }

        intervals.clear();
        string line;
        int line_no = 0;
        while (getline(in, line)) {
            ++line_no;
            size_t hash = line.find('#');
            if (hash != string::npos) line = line.substr(0, hash);
            istringstream is(line);
            double t0=0.0, t1=0.0, intensity_mmh=0.0;
            if (!(is >> t0 >> t1 >> intensity_mmh)) continue;
            if (t1 <= t0) {
                cerr<<"[WARN] skip rain interval with t1<=t0 at "<<path<<":"<<line_no<<"\n";
                continue;
            }
            if (intensity_mmh < 0.0) {
                cerr<<"[WARN] negative rain intensity clipped to 0 at "<<path<<":"<<line_no<<"\n";
                intensity_mmh = 0.0;
            }
            intervals.push_back({t0, t1, intensity_mmh / (1000.0 * 3600.0)});
        }

        sort(intervals.begin(), intervals.end(), [](const RainInterval& a, const RainInterval& b){
            return a.t0 < b.t0;
        });

        H = 0.0;
        T = 0.0;
        for (const auto& it: intervals) {
            H += it.rate * (it.t1 - it.t0);
            T = max(T, it.t1);
        }
        mode = RainMode::FILE;
        r_uniform = 0.0;
        if (intervals.empty()) cerr<<"[WARN] rain file has no valid intervals: "<<path<<"\n";
        return !intervals.empty();
    }

    double rate(double t) const { // m/s
        if (t<0 || t>=T) return 0.0;
        if (mode==RainMode::UNIFORM) return r_uniform;
        if (mode==RainMode::FILE) {
            for (const auto& it: intervals) {
                if (t>=it.t0 && t<it.t1) return it.rate;
            }
            return 0.0;
        }
        for (const auto& p: pulses){
            if (t>=p.t0 && t<p.t1){
                double dur = max(1e-9, p.t1-p.t0);
                return (H*p.frac)/dur;
            }
        }
        return 0.0;
    }
};

// -------------- Horton 下渗 --------------
struct HortonConfig {
    bool enabled = false;
    double f0 = 0.0;  // 初始下渗能力 (m/s)
    double fc = 0.0;  // 稳定下渗能力 (m/s)
    double k  = 0.0;  // 衰减系数 (1/s)
    double t_ref = 0.0;

    void normalize() {
        if (f0 < fc) swap(f0, fc);
        if (fc < 0.0) fc = 0.0;
        if (k  < 0.0) k  = 0.0;
    }
    double capacity(double wet_time) const {
        if (!enabled) return 0.0;
        const double te = max(0.0, wet_time);
        return fc + (f0 - fc) * exp(-k * te);
    }
};

inline double manning_denominator(double ht, double n, double q_abs, double gdt){
    // 等价于 ht^(7/3)=ht^2*cbrt(ht)，比 pow(ht, 2.333333) 更快
    const double ht_7_3 = ht * ht * cbrt(ht);
    return 1.0 + gdt * n * n * q_abs / ht_7_3;
}

inline void apply_rain_and_infiltration(
    Hydro_cell **Grid, int Row, int Col, double rain_rate, double time_now, double dt, const HortonConfig& horton
){
    (void)time_now;
    const double rain_depth = rain_rate * dt;
    if (!horton.enabled){
        #pragma omp parallel for schedule(static)
        for (int i=0; i<Row; ++i){
            for (int j=0; j<Col; ++j){
                if (Grid[i][j].noData) continue;
                Grid[i][j].Storage += rain_depth;
                Grid[i][j].Absolute_h = Grid[i][j].Dem + Grid[i][j].Storage;
            }
        }
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i=0; i<Row; ++i){
        for (int j=0; j<Col; ++j){
            if (Grid[i][j].noData) continue;
            double s = Grid[i][j].Storage + rain_depth;
            if (s > 0.0){
                const double infil_depth_cap = horton.capacity(Grid[i][j].wet_elapsed) * dt;
                s -= min(infil_depth_cap, s);
                Grid[i][j].wet_elapsed += dt;
            }
            Grid[i][j].Storage = s;
            Grid[i][j].Absolute_h = Grid[i][j].Dem + s;
        }
    }
}

// -------------- 函数声明 --------------
void Paul_bates_inertial(Hydro_cell **Grid, int Row, int Col, float res, float dt);

// -------------- 主程序 --------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    // Do not force flushing after every cout operation. Per-step flushing is
    // expensive when adaptive dt is small; important status lines still flush
    // at program exit or when the shell closes the redirected log.

    int Row=1077, Col=1517;
    string info_input_file="./INPUT_INFO";
    ifstream infile(info_input_file.c_str());
    if(!infile){ cerr<<"Cannot open inputfile "<<info_input_file<<"\n"; return 1; }

    // 读取控制参数
    string str, k, v;
    int total_time=3600;
    string outfile_path="NONE", storagefile_path="NONE";
    string dsm_file_path="NONE", landcover_file_path="NONE";
    string out_fdr_path="process/";
    string rain_mode_str="UNIFORM";
    string rain_file_path="NONE";
    string roughness_mode = "LANDCOVER"; // LANDCOVER / UNIFORM
    double H_total = 0.0;       // WATER_DEPTH(m) 作为总雨深
    float  dt_user = 1.0f;
    int    out_step = 60;
    double log_out_step = 60.0;
    string infiltration_switch = "OFF";
    double horton_f0_mmh = 0.0, horton_fc_mmh = 0.0, horton_k_per_h = 0.0;
    float  manning_n_uniform = 0.025f; // 全域统一裸土(壤土)默认值
    OPTION options;

    while (getline(infile, str)) {
        if (str.empty()) continue;
        istringstream is(str); is >> k >> v;
        if (k=="ROW") Row = StringToNum<int>(v);
        else if (k=="COL") Col = StringToNum<int>(v);
        else if (k=="TOTAL_TIME(s)") total_time = StringToNum<int>(v);
        else if (k=="DSM_PATH") dsm_file_path = v;
        else if (k=="LANDCOVER_PATH") landcover_file_path = v;
        else if (k=="OUTPUT_PATH") outfile_path = v;
        else if (k=="STORAGE_PATH") storagefile_path = v;
        else if (k=="WATER_DEPTH(m)") H_total = StringToNum<double>(v);           // ← 总雨深（米）
        else if (k=="TIME_STEP(s)") dt_user = StringToNum<float>(v);
        else if (k=="TIME_ADAPTIVE") options.time_adaptive = v;
        else if (k=="RESOLUTION") options.res = StringToNum<double>(v);
        else if (k=="FDR_OUTSTEP") out_step = StringToNum<int>(v);
        else if (k=="LOG_OUTSTEP(s)" || k=="LOG_OUTSTEP") log_out_step = StringToNum<double>(v);
        else if (k=="FDR_PATH") out_fdr_path = v;
        else if (k=="RAIN_MODE") rain_mode_str = v;
        else if (k=="RAIN_FILE_PATH") rain_file_path = v;
        else if (k=="ROUGHNESS_MODE") roughness_mode = v;
        else if (k=="INFILTRATION") infiltration_switch = v;
        else if (k=="HORTON_F0(mm/h)") horton_f0_mmh = StringToNum<double>(v);
        else if (k=="HORTON_FC(mm/h)") horton_fc_mmh = StringToNum<double>(v);
        else if (k=="HORTON_K(1/h)") horton_k_per_h = StringToNum<double>(v);
        else if (k=="MANNING_N_UNIFORM") manning_n_uniform = StringToNum<float>(v);
    }
    infile.close();

    if (H_total > 1.0)
        cerr<<"[WARN] WATER_DEPTH(m) = "<<H_total<<" looks large. Did you mean meters (e.g. 20mm -> 0.02)?\n";

    cout<<"**** Time step read: "<<dt_user<<" s ; Total rainfall H: "<<H_total<<" m ****\n";
    cout<<"**** Time Adaptive: "<<options.time_adaptive<<" ; Resolution: "<<options.res<<" m ****\n";

    HortonConfig horton;
    horton.enabled = (infiltration_switch=="ON" || infiltration_switch=="on" || infiltration_switch=="1");
    horton.f0 = horton_f0_mmh / (1000.0 * 3600.0); // mm/h -> m/s
    horton.fc = horton_fc_mmh / (1000.0 * 3600.0); // mm/h -> m/s
    horton.k  = horton_k_per_h / 3600.0;           // 1/h  -> 1/s
    horton.normalize();
    if (horton.enabled){
        cout<<"**** Horton ON: f0="<<horton_f0_mmh<<" mm/h, fc="<<horton_fc_mmh<<" mm/h, k="<<horton_k_per_h<<" 1/h ****\n";
    } else {
        cout<<"**** Horton OFF ****\n";
    }
    bool use_landcover = (roughness_mode=="LANDCOVER" || roughness_mode=="landcover" || roughness_mode=="LC" || roughness_mode=="lc");
    if (use_landcover) cout<<"**** Roughness mode: LANDCOVER ****\n";
    else               cout<<"**** Roughness mode: UNIFORM n="<<manning_n_uniform<<" ****\n";

    // 分配网格
    Hydro_cell **Grid = new Hydro_cell*[Row];
    for (int i=0;i<Row;++i) Grid[i] = new Hydro_cell[Col];

    // 读取 DEM / LC
    ifstream indem(dsm_file_path.c_str());
    if(!indem){ cerr<<"cannot open dem file: "<<dsm_file_path<<"\n"; return 1; }
    ifstream inland;
    if (use_landcover){
        inland.open(landcover_file_path.c_str());
        if(!inland){
            cerr<<"[WARN] cannot open landcover: "<<landcover_file_path<<", fallback to UNIFORM n="<<manning_n_uniform<<"\n";
            use_landcover = false;
        }
    }

    string head[6];
    for (int i=0;i<6;i++){
        getline(indem, head[i]);
        if (use_landcover) getline(inland, str);
    }

    ifstream inStorage(storagefile_path.c_str());
    if(inStorage){ for (int i=0;i<6;i++) getline(inStorage, str); }
    else { cout<<"**** No initial storage file. Start DRY (Storage=0) ****\n"; }

    // 填充格网（初始干燥）
    double tmp; int lc=0;
    for (int i=0;i<Row;i++){
        for (int j=0;j<Col;j++){
            Grid[i][j].noData = 0;
            indem >> tmp; Grid[i][j].Dem = tmp;

            if (inStorage) { inStorage >> tmp; Grid[i][j].Storage = tmp; }
            else           { Grid[i][j].Storage = 0.0; }    // 初始干燥

            Grid[i][j].Absolute_h = Grid[i][j].Dem + Grid[i][j].Storage;

            if (use_landcover){
                inland >> lc;
                Grid[i][j].landcover = lc;
                switch(lc){
                    case 1:  Grid[i][j].n = 0.035f; break;
                    case 2:  Grid[i][j].n = 0.4f;   break;
                    case 3:  Grid[i][j].n = 0.15f;  break;
                    case 4:  Grid[i][j].n = 0.035f; break;
                    case 5:  Grid[i][j].n = 0.17f;  break;
                    case 6:  Grid[i][j].n = 0.8f;   break;
                    case 7:  Grid[i][j].n = 0.012f; break;
                    case 8:  Grid[i][j].n = 0.013f; break;
                    case 9:  Grid[i][j].n = 0.035f; break;
                    case 10: Grid[i][j].n = 0.035f; break;
                    default: Grid[i][j].n = 0.035f; break;
                }
            } else {
                Grid[i][j].n = manning_n_uniform;
            }

            if (Grid[i][j].Dem <= -999) Grid[i][j].noData = 1;
        }
    }
    indem.close(); if(inland) inland.close(); if(inStorage) inStorage.close();

    // 邻界 emax
    for (int i=0;i<Row;i++) for (int j=0;j<Col;j++){
        if (i>0)       Grid[i][j].emax_up    = max(Grid[i][j].Dem, Grid[i-1][j].Dem);
        if (i<Row-1)   Grid[i][j].emax_down  = max(Grid[i][j].Dem, Grid[i+1][j].Dem);
        if (j>0)       Grid[i][j].emax_left  = max(Grid[i][j].Dem, Grid[i][j-1].Dem);
        if (j<Col-1)   Grid[i][j].emax_right = max(Grid[i][j].Dem, Grid[i][j+1].Dem);
    }

    cout<<"**** Initialize over, now Run inundation ****\n";
    clock_t start = clock();

    // 降雨调度
    RainMode rmode = RainMode::UNIFORM;
    if (rain_mode_str=="FRONT") rmode = RainMode::FRONT;
    else if (rain_mode_str=="BACK") rmode = RainMode::BACK;
    else if (rain_mode_str=="FILE" || rain_mode_str=="EXTERNAL" || rain_mode_str=="REAL") rmode = RainMode::FILE;
    RainScheduler rain;
    if (rmode == RainMode::FILE) {
        if (rain_file_path=="NONE" || rain_file_path.empty()) {
            cerr<<"**** Error: RAIN_MODE=FILE needs RAIN_FILE_PATH ****\n";
            return 1;
        }
        if (!rain.load_file(rain_file_path)) return 1;
        cout<<"**** Rain mode: FILE ; path="<<rain_file_path
            <<" ; file duration="<<rain.T<<" s ; file depth="<<(rain.H*1000.0)<<" mm ****\n";
        if (H_total > 0.0 && fabs(H_total - rain.H) > 1e-6) {
            cerr<<"[WARN] WATER_DEPTH(m)="<<H_total<<" differs from rain file depth="<<rain.H
                <<". FILE rates are used.\n";
        }
    } else {
        rain.init(rmode, H_total, total_time);
    }

    // 输出目录
    if (!out_fdr_path.empty()){
        // 尽量创建；失败也不退出
        std::system( (string("mkdir -p \"") + out_fdr_path + "\"").c_str() );
    }

    // 主循环
    if (options.time_adaptive=="OFF"){
        int total_steps = int(total_time / dt_user);
        int num_out = 1;
        for (int t=1; t<=total_steps; ++t){
            double time_now = (t-1)*dt_user;

            // 先加雨
            double r = rain.rate(time_now); // m/s
            apply_rain_and_infiltration(Grid, Row, Col, r, time_now, dt_user, horton);

            // 水动力
            Paul_bates_inertial(Grid, Row, Col, options.res, dt_user);

            // 过程输出
            double tsec = t*dt_user;
            if (out_step>0 && tsec+1e-9 >= num_out*out_step){
                string out_pro_file = out_fdr_path + toString(num_out*out_step) + "s.asc";
                ofstream out_pro(out_pro_file.c_str());
                if (out_pro){
                    for (int k=0;k<6;k++) out_pro<<head[k]<<"\n";
                    for (int i=0;i<Row;i++){
                        for (int j=0;j<Col;j++){
                            if (Grid[i][j].noData || Grid[i][j].Storage<=1e-6) out_pro<<"0 ";
                            else out_pro << (Grid[i][j].Storage*1000.0) << " ";
                        }
                        out_pro<<"\n";
                    }
                    out_pro.close();
                }
                ++num_out;
            }
        }
    } else if (options.time_adaptive=="ON"){
        double time_now=0.0;
        const double cfl=0.2, g=9.81;
        int num_out=1;
        double next_log_time = 0.0;

        while (time_now < total_time-1e-9){
            // 估计 hmax
            double hmax=0.0;
            #pragma omp parallel for reduction(max:hmax)
            for (int i=0;i<Row;i++) for (int j=0;j<Col;j++)
                if (!Grid[i][j].noData) hmax = max(hmax, Grid[i][j].Storage);

            float dt = dt_user;
            if (hmax>=1e-8){
                dt = float( cfl * options.res / sqrt(g * hmax) );
                if (dt > dt_user) dt = dt_user;
            }
            if (time_now + dt > total_time) dt = float(total_time - time_now);
            if (dt <= 0) break;

            // 加雨
            double r = rain.rate(time_now); // m/s
            apply_rain_and_infiltration(Grid, Row, Col, r, time_now, dt, horton);

            // 水动力
            Paul_bates_inertial(Grid, Row, Col, options.res, dt);
            time_now += dt;
            if (log_out_step > 0.0 && time_now + 1e-9 >= next_log_time) {
                cout<<"**** Time elapsed: "<<time_now<<" s ; dt="<<dt<<" ****\n";
                next_log_time += log_out_step;
                if (next_log_time < time_now) next_log_time = time_now + log_out_step;
            }

            // 过程输出
            if (out_step>0 && time_now+1e-6 >= num_out*out_step){
                string out_pro_file = out_fdr_path + toString(num_out*out_step) + "s.asc";
                ofstream out_pro(out_pro_file.c_str());
                if (out_pro){
                    for (int k=0;k<6;k++) out_pro<<head[k]<<"\n";
                    for (int i=0;i<Row;i++){
                        for (int j=0;j<Col;j++){
                            if (Grid[i][j].noData || Grid[i][j].Storage<=1e-6) out_pro<<"0 ";
                            else out_pro << (Grid[i][j].Storage*1000.0) << " ";
                        }
                        out_pro<<"\n";
                    }
                    out_pro.close();
                }
                ++num_out;
            }
        }
    } else {
        cerr<<"**** Error: TIME_ADAPTIVE must be ON or OFF (now "<<options.time_adaptive<<") ****\n";
        return 1;
    }

    clock_t finish = clock();
    cout<<"**** Time consuming: "<< double(finish - start)/CLOCKS_PER_SEC <<" (s) ****\n";

    // 最终输出（mm）
    ofstream outfile(outfile_path.c_str());
    if (!outfile) cerr<<"cannot open: "<<outfile_path<<"\n";
    else{
        for (int k=0;k<6;k++) outfile<<head[k]<<"\n";
        for (int i=0;i<Row;i++){
            for (int j=0;j<Col;j++){
                if (Grid[i][j].noData || Grid[i][j].Storage<=1e-6) outfile<<"0 ";
                else outfile<<(Grid[i][j].Storage*1000.0)<<" ";
            }
            outfile<<"\n";
        }
        outfile.close();
        cout<<"Wrote final grid (mm): "<<outfile_path<<"\n";
    }

    // 释放
    for (int i=0;i<Row;i++) delete[] Grid[i];
    delete[] Grid;

    cout<<"**** Simulation Finished ****\n";
    return 0;
}

// ----------------- 水动力核心（保持与原始一致；仅对安全环节并行） -----------------
void Paul_bates_inertial(Hydro_cell **Grid, int Row, int Col, float res, float dt)
{
    int i, j;
    const double dx=res, dy=res;
    const double g=9.81;

    double dh, ht, dwdx, dwdy, q_last;
    double qx_left, qx_right, qy_up, qy_down;

    // 1) 预测各向通量（每格只写自己字段，可并行）
    #pragma omp parallel for collapse(2) private(qx_left,qx_right,qy_up,qy_down,ht,dwdx,dwdy,q_last,dh)
    for (i=0; i<Row; i++){
        for (j=0; j<Col; j++){
            if (Grid[i][j].noData==1){
                Grid[i][j].q_up_now=Grid[i][j].q_botton_now=Grid[i][j].q_left_now=Grid[i][j].q_right_now=0.0;
                Grid[i][j].dh=0.0; continue;
            }
            qy_up=0;
            if (i-1>=0 && Grid[i-1][j].noData!=1){
                if (Grid[i][j].Absolute_h > Grid[i-1][j].Absolute_h){
                    dwdy = -(Grid[i][j].Absolute_h - Grid[i-1][j].Absolute_h)/dy;
                    ht = Grid[i][j].Storage + Grid[i][j].Dem - Grid[i][j].emax_up;
                    q_last = Grid[i][j].q_up_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qy_up = (q_last - g*ht*dt*dwdy) / (1 + g*ht*dt*Grid[i][j].n*Grid[i][j].n*fabs(q_last)/pow(ht,3.333333));
                    }
                }else{
                    dwdy = -(Grid[i-1][j].Absolute_h - Grid[i][j].Absolute_h)/dy;
                    ht = Grid[i-1][j].Storage + Grid[i-1][j].Dem - Grid[i][j].emax_up;
                    q_last = Grid[i-1][j].q_botton_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qy_up = (q_last - g*ht*dt*dwdy) / (1 + g*ht*dt*Grid[i-1][j].n*Grid[i-1][j].n*fabs(q_last)/pow(ht,3.333333));
                        qy_up = -qy_up;
                    }
                }
            }
            qy_down=0;
            if (i+1<=Row-1 && Grid[i+1][j].noData!=1){
                if (Grid[i][j].Absolute_h > Grid[i+1][j].Absolute_h){
                    dwdy = -(Grid[i][j].Absolute_h - Grid[i+1][j].Absolute_h)/dy;
                    ht = Grid[i][j].Storage + Grid[i][j].Dem - Grid[i][j].emax_down;
                    q_last = Grid[i][j].q_botton_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qy_down = (q_last - g*ht*dt*dwdy) / (1 + g*ht*dt*Grid[i][j].n*Grid[i][j].n*fabs(q_last)/pow(ht,3.333333));
                    }
                }else{
                    dwdy = -(Grid[i+1][j].Absolute_h - Grid[i][j].Absolute_h)/dy;
                    ht = Grid[i+1][j].Storage + Grid[i+1][j].Dem - Grid[i][j].emax_down;
                    q_last = Grid[i+1][j].q_up_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qy_down = (q_last - g*ht*dt*dwdy) / (1 + g*ht*dt*Grid[i+1][j].n*Grid[i+1][j].n*fabs(q_last)/pow(ht,3.333333));
                        qy_down = -qy_down;
                    }
                }
            }
            qx_left=0;
            if (j-1>=0 && Grid[i][j-1].noData!=1){
                if (Grid[i][j].Absolute_h > Grid[i][j-1].Absolute_h){
                    dwdx = -(Grid[i][j].Absolute_h - Grid[i][j-1].Absolute_h)/dx;
                    ht = Grid[i][j].Storage + Grid[i][j].Dem - Grid[i][j].emax_left;
                    q_last = Grid[i][j].q_left_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qx_left = (q_last - g*ht*dt*dwdx) / (1 + g*ht*dt*Grid[i][j].n*Grid[i][j].n*fabs(q_last)/pow(ht,3.333333));
                    }
                }else{
                    dwdx = -(Grid[i][j-1].Absolute_h - Grid[i][j].Absolute_h)/dx;
                    ht = Grid[i][j-1].Storage + Grid[i][j-1].Dem - Grid[i][j].emax_left;
                    q_last = Grid[i][j-1].q_right_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qx_left = (q_last - g*ht*dt*dwdx) / (1 + g*ht*dt*Grid[i][j-1].n*Grid[i][j-1].n*fabs(q_last)/pow(ht,3.333333));
                        qx_left = -qx_left;
                    }
                }
            }
            qx_right=0;
            if (j+1<=Col-1 && Grid[i][j+1].noData!=1){
                if (Grid[i][j].Absolute_h > Grid[i][j+1].Absolute_h){
                    dwdx = -(Grid[i][j].Absolute_h - Grid[i][j+1].Absolute_h)/dx;
                    ht = Grid[i][j].Storage + Grid[i][j].Dem - Grid[i][j].emax_right;
                    q_last = Grid[i][j].q_right_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qx_right = (q_last - g*ht*dt*dwdx) / (1 + g*ht*dt*Grid[i][j].n*Grid[i][j].n*fabs(q_last)/pow(ht,3.333333));
                    }
                }else{
                    dwdx = -(Grid[i][j+1].Absolute_h - Grid[i][j].Absolute_h)/dx;
                    ht = Grid[i][j+1].Storage + Grid[i][j+1].Dem - Grid[i][j].emax_right;
                    q_last = Grid[i][j+1].q_left_last;
                    if (!(fabs(ht)<1e-8 || ht<0)){
                        qx_right = (q_last - g*ht*dt*dwdx) / (1 + g*ht*dt*Grid[i][j+1].n*Grid[i][j+1].n*fabs(q_last)/pow(ht,3.333333));
                        qx_right = -qx_right;
                    }
                }
            }

            Grid[i][j].q_up_now = qy_up;
            Grid[i][j].q_botton_now = qy_down;
            Grid[i][j].q_left_now = qx_left;
            Grid[i][j].q_right_now = qx_right;
            // 先不更新 dh，等 LIMITER 之后再统一计算
        }
    }

#if LIMITER
    // === 原版 LIMITER：保持顺序执行以确保结果一致 ===
    float dh2, dh3;
    for (i = 0; i < Row; i++){
        for (j = 0; j < Col; j++){
            if (Grid[i][j].noData==1) continue;

            float q_out_max = float( max(0.0, Grid[i][j].Storage) * dx * dy / max(double(dt),1e-9) / 2.0 );
            float qtmp[4];
            qtmp[0] = float(Grid[i][j].q_up_now);
            qtmp[1] = float(Grid[i][j].q_botton_now);
            qtmp[2] = float(Grid[i][j].q_left_now);
            qtmp[3] = float(Grid[i][j].q_right_now);

            int ii;
            float minus_q = 0;
            for (ii = 0; ii < 4; ii++)
                if (qtmp[ii] > 0) minus_q += qtmp[ii];

            double dh_local = -minus_q / dx / dy * dt;
            if (dh_local + Grid[i][j].Storage < 0){
                float qratio;
                if (Grid[i][j].q_up_now > 0){
                    if (i-1>=0 && fabs(Grid[i][j].Absolute_h - Grid[i-1][j].Absolute_h) < 1e-10){
                        Grid[i][j].q_up_now = 0; Grid[i-1][j].q_botton_now = 0;
                    } else if (i-1>=0){
                        qratio = float(Grid[i][j].q_up_now / minus_q);
                        Grid[i][j].q_up_now = qratio * q_out_max;
                        Grid[i-1][j].q_botton_now = -Grid[i][j].q_up_now;
                        dh2 = -Grid[i-1][j].q_botton_now / dx / dy * dt;
#if LIMIT_Q
                        if (dh2 > Grid[i][j].Absolute_h - Grid[i-1][j].Absolute_h){
                            dh3 = float(Grid[i][j].Absolute_h - Grid[i-1][j].Absolute_h);
                            Grid[i][j].q_up_now = dh3 * dx * dy / dt;
                            Grid[i-1][j].q_botton_now = -Grid[i][j].q_up_now;
                        }
#endif
                    }
                }
                if (Grid[i][j].q_botton_now > 0){
                    if (i+1<=Row-1 && fabs(Grid[i][j].Absolute_h - Grid[i+1][j].Absolute_h) < 1e-10){
                        Grid[i][j].q_botton_now = 0; Grid[i+1][j].q_up_now = 0;
                    } else if (i+1<=Row-1){
                        qratio = float(Grid[i][j].q_botton_now / minus_q);
                        Grid[i][j].q_botton_now = qratio * q_out_max;
                        Grid[i+1][j].q_up_now = -Grid[i][j].q_botton_now;
                        dh2 = Grid[i+1][j].q_up_now / dx / dy * dt;
#if LIMIT_Q
                        if (dh2 > Grid[i][j].Absolute_h - Grid[i+1][j].Absolute_h){
                            dh3 = float(Grid[i][j].Absolute_h - Grid[i+1][j].Absolute_h);
                            Grid[i][j].q_botton_now = dh3 * dx * dy / dt;
                            Grid[i+1][j].q_up_now = -Grid[i][j].q_botton_now;
                        }
#endif
                    }
                }
                if (Grid[i][j].q_left_now > 0){
                    if (j-1>=0 && fabs(Grid[i][j].Absolute_h - Grid[i][j-1].Absolute_h) < 1e-10){
                        Grid[i][j].q_left_now = 0; Grid[i][j-1].q_right_now = 0;
                    } else if (j-1>=0){
                        qratio = float(Grid[i][j].q_left_now / minus_q);
                        Grid[i][j].q_left_now = qratio * q_out_max;
                        Grid[i][j-1].q_right_now = -Grid[i][j].q_left_now;
                        dh2 = Grid[i][j-1].q_right_now / dx / dy * dt;
#if LIMIT_Q
                        if (dh2 > Grid[i][j].Absolute_h - Grid[i][j-1].Absolute_h){
                            dh3 = float(Grid[i][j].Absolute_h - Grid[i][j-1].Absolute_h);
                            Grid[i][j].q_left_now = dh3 * dx * dy / dt;
                            Grid[i][j-1].q_right_now = -Grid[i][j].q_left_now;
                        }
#endif
                    }
                }
                if (Grid[i][j].q_right_now > 0){
                    if (j+1<=Col-1 && fabs(Grid[i][j].Absolute_h - Grid[i][j+1].Absolute_h) < 1e-10){
                        Grid[i][j].q_right_now = 0; Grid[i][j+1].q_left_now = 0;
                    } else if (j+1<=Col-1){
                        qratio = float(Grid[i][j].q_right_now / minus_q);
                        Grid[i][j].q_right_now = qratio * q_out_max;
                        Grid[i][j+1].q_left_now = -Grid[i][j].q_right_now;
                        dh2 = Grid[i][j+1].q_left_now / dx / dy * dt;
#if LIMIT_Q
                        if (dh2 > Grid[i][j].Absolute_h - Grid[i][j+1].Absolute_h){
                            dh3 = float(Grid[i][j].Absolute_h - Grid[i][j+1].Absolute_h);
                            Grid[i][j].q_right_now = dh3 * dx * dy / dt;
                            Grid[i][j+1].q_left_now = -Grid[i][j].q_right_now;
                        }
#endif
                    }
                }
            }
        }
    }
#endif // LIMITER

    // 2) 统一计算 dh（可并行）
    #pragma omp parallel for collapse(2)
    for (i=0; i<Row; i++){
        for (j=0; j<Col; j++){
            if (Grid[i][j].noData==1){ Grid[i][j].dh=0.0; continue; }
            Grid[i][j].dh = -(Grid[i][j].q_left_now + Grid[i][j].q_right_now + Grid[i][j].q_up_now + Grid[i][j].q_botton_now) / (dx*dy) * dt;
        }
    }

    // 3) 更新（可并行）
    #pragma omp parallel for collapse(2)
    for (i=0; i<Row; i++){
        for (j=0; j<Col; j++){
            if (Grid[i][j].noData==1) continue;
            Grid[i][j].q_right_last = Grid[i][j].q_right_now;
            Grid[i][j].q_left_last  = Grid[i][j].q_left_now;
            Grid[i][j].q_up_last    = Grid[i][j].q_up_now;
            Grid[i][j].q_botton_last= Grid[i][j].q_botton_now;

            Grid[i][j].Storage += Grid[i][j].dh;
            if (Grid[i][j].Storage < 0.0) Grid[i][j].Storage = 0.0; // 裁剪负水深（避免数值误差）
            Grid[i][j].Absolute_h = Grid[i][j].Storage + Grid[i][j].Dem;
        }
    }
}
