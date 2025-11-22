import os
import sys
import shutil
import glob
import subprocess
import numpy as np
import pybind11

def setup_lanms() -> bool:
    # Clean up previous builds and imports (if needed)
    if 'lanms' in sys.modules: 
        del sys.modules['lanms']
    if 'lanms_cpu' in sys.modules: 
        del sys.modules['lanms_cpu']
    
    for d in ['lanms_src', 'build', 'temp_pyclipper']:
        if os.path.exists(d): 
            shutil.rmtree(d)
    for f in glob.glob("*.so"): 
        os.remove(f)
    if os.path.exists("lanms.py"): 
        os.remove("lanms.py")
    if os.path.exists("setup.py"): 
        os.remove("setup.py")

    os.makedirs('lanms_src', exist_ok=True)

    # Download Clipper from pyclipper repo
    print("Downloading Clipper...")
    try:
        subprocess.run(["git", "clone", "https://github.com/fonttools/pyclipper.git", "temp_pyclipper"], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        found = False
        for root, _, files in os.walk("temp_pyclipper"):
            if "clipper.cpp" in files and "clipper.hpp" in files:
                shutil.copy(os.path.join(root, "clipper.cpp"), "lanms_src/clipper.cpp")
                shutil.copy(os.path.join(root, "clipper.hpp"), "lanms_src/clipper.hpp")
                found = True
                break
        if not found:
            print("Error: Clipper files not found in cloned repo")
            return False
    except Exception as e:
        print(f"Git clone failed: {e}")
        return False
    finally:
        if os.path.exists("temp_pyclipper"): 
            shutil.rmtree("temp_pyclipper")

    # Create lanms.h with the NMS implementation
    print("Creating lanms.h...")
    lanms_h_code = r"""
#pragma once
#include "clipper.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace lanms {
    namespace cl = ClipperLib;
    struct Polygon { cl::Path poly; float score; };

    inline float poly_iou(const Polygon &a, const Polygon &b) {
        cl::Clipper clpr;
        clpr.AddPath(a.poly, cl::ptSubject, true);
        clpr.AddPath(b.poly, cl::ptClip, true);
        cl::Paths inter;
        clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd, cl::pftEvenOdd);
        if (inter.size() == 0) return 0;
        
        float inter_area = 0;
        for (auto &&p : inter) inter_area += cl::Area(p);
        
        float a_area = cl::Area(a.poly);
        float b_area = cl::Area(b.poly);
        float union_area = a_area + b_area - inter_area;
        if (union_area <= 0) return 0;
        return inter_area / union_area;
    }

    inline std::vector<std::vector<float>> merge_quadrangle_n9(const std::vector<float>& data, float iou_threshold) {
        using namespace ClipperLib;
        std::vector<Polygon> polys;
        for (size_t i = 0; i < data.size(); i += 9) {
            Polygon p;
            p.score = data[i + 8];
            for (int j = 0; j < 4; j++) {
                p.poly.push_back(cl::IntPoint((long long)(data[i + j * 2] * 10000), (long long)(data[i + j * 2 + 1] * 10000)));
            }
            polys.push_back(p);
        }

        std::vector<int> merged(polys.size(), 0);
        std::vector<std::vector<float>> res;
        std::vector<size_t> indices(polys.size());
        for(size_t i=0; i<indices.size(); ++i) indices[i] = i;
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return polys[i].score > polys[j].score; });

        for (size_t i : indices) {
            if (merged[i]) continue;
            Polygon p_merged = polys[i];
            float weight_sum = p_merged.score;
            double mx[4] = {0}, my[4] = {0};
            
            for(int k=0; k<4; k++) {
                mx[k] = (double)p_merged.poly[k].X * weight_sum;
                my[k] = (double)p_merged.poly[k].Y * weight_sum;
            }

            for (size_t j : indices) {
                if (i == j || merged[j]) continue;
                if (poly_iou(polys[i], polys[j]) > iou_threshold) {
                    float w = polys[j].score;
                    for(int k=0; k<4; k++) {
                        mx[k] += (double)polys[j].poly[k].X * w;
                        my[k] += (double)polys[j].poly[k].Y * w;
                    }
                    weight_sum += w;
                    merged[j] = 1;
                }
            }
            std::vector<float> r;
            for(int k=0; k<4; k++) {
                r.push_back((float)(mx[k] / weight_sum / 10000.0));
                r.push_back((float)(my[k] / weight_sum / 10000.0));
            }
            r.push_back(p_merged.score);
            res.push_back(r);
        }
        return res;
    }
}
"""
    with open("lanms_src/lanms.h", "w") as f: 
        f.write(lanms_h_code)

    # Create adaptor.cpp to wrap the NMS function
    print("Creating adaptor.cpp...")
    adaptor_code = """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "lanms.h"

namespace py = pybind11;

std::vector<std::vector<float>> merge_wrapper(
    py::array_t<float, py::array::c_style | py::array::forcecast> quad_n9,
    float iou_threshold) {
    
    auto buf = quad_n9.request();
    float* ptr = (float*)buf.ptr;
    int N = buf.shape[0];
    std::vector<float> data;
    data.assign(ptr, ptr + N * 9);
    return lanms::merge_quadrangle_n9(data, iou_threshold);
}

// CHÚ Ý: Tên module ở đây là 'lanms_cpu'
PYBIND11_MODULE(lanms_cpu, m) {
    m.doc() = "Locality-Aware NMS (Renamed Version)";
    m.def("merge_quadrangle_n9", &merge_wrapper, "Locality-Aware NMS");
}
"""
    with open("lanms_src/adaptor.cpp", "w") as f: 
        f.write(adaptor_code)

    # Setup file to build the C++ extension
    print("Compiling 'lanms_cpu' extension...")
    setup_code = f"""
from setuptools import setup, Extension
import numpy as np
import pybind11

ext = Extension('lanms_cpu', 
    sources=['lanms_src/adaptor.cpp', 'lanms_src/clipper.cpp'],
    include_dirs=[np.get_include(), 'lanms_src', '{pybind11.get_include()}'],
    extra_compile_args=['-std=c++11'],
    language='c++'
)
setup(name='lanms_cpu', ext_modules=[ext])
"""
    with open("setup.py", "w") as f: 
        f.write(setup_code)

    try:
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("Failed to build lanms_cpu extension")
        return False

    # Create lanms.py wrapper
    print("Creating wrapper 'lanms.py'...")
    with open("lanms.py", "w") as f:
        f.write("try:\n")
        f.write("    # Import extension C++ lanms_cpu\n")
        f.write("    from lanms_cpu import *\n")
        f.write("    print('C++ NMS loaded via lanms_cpu')\n")
        f.write("except ImportError as e:\n")
        f.write("    print(f'Wrapper Import Failed: {e}')\n")

    return True

if __name__ == "__main__":
    if setup_lanms():
        print("\nBuild successful")
        # Check .so file
        so = glob.glob("lanms_cpu*.so")
        print(f"Shared Object: {so[0] if so else 'Missing'}")
        
        # Test import
        print("\nTesting Import...")
        try:
            import lanms
            if hasattr(lanms, 'merge_quadrangle_n9'):
                print("Successfully imported 'lanms' module. Function found")
                d = np.array([
                    [10, 10, 20, 10, 20, 20, 10, 20, 0.9],
                    [12, 12, 22, 12, 22, 22, 12, 22, 0.8],
                    [30, 30, 40, 30, 40, 40, 30, 40, 0.7],
                    [32, 32, 42, 32, 42, 42, 32, 42, 0.6]
                ], dtype=np.float32)
                res = lanms.merge_quadrangle_n9(d, 0.5)
                print(f"Success test execution: {res})")
            else:
                print("Import succeeded but function not found")
        except ImportError as e:
            print(f"Import Error: {e}")