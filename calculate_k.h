#ifndef K_INTERPOLATION_H
#define K_INTERPOLATION_H

#include <string>
#include <vector>
#include <H5Cpp.h>
#include <algorithm>
#include <iostream>

// Function to load `t` (1/t) and `k` (1/v1r) data from an HDF5 file
inline void load_k_data(const std::string& filename, std::vector<double>& t_data, std::vector<double>& k_data) {
    try {
        // Open the HDF5 file
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // Read the `1/t` dataset
        H5::DataSet t_dataset = file.openDataSet("1/t");
        H5::DataSpace t_dataspace = t_dataset.getSpace();
        hsize_t t_size;
        t_dataspace.getSimpleExtentDims(&t_size, nullptr);
        t_data.resize(t_size);
        t_dataset.read(t_data.data(), H5::PredType::NATIVE_DOUBLE);

        // Read the `1/v1r` dataset
        H5::DataSet k_dataset = file.openDataSet("1/v3r");
        H5::DataSpace k_dataspace = k_dataset.getSpace();
        hsize_t k_size;
        k_dataspace.getSimpleExtentDims(&k_size, nullptr);
        k_data.resize(k_size);
        k_dataset.read(k_data.data(), H5::PredType::NATIVE_DOUBLE);

        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "Error reading HDF5 file: " << e.getDetailMsg() << std::endl;
        throw;
    }
}

// Function to interpolate `k` for a given `t` using linear interpolation
inline double interpolate_k(double t, const std::vector<double>& t_data, const std::vector<double>& k_data) {
    // Ensure t is within bounds
    if (t <= t_data.front()) return k_data.front();
    if (t >= t_data.back()) return k_data.back();

    // Find the interval containing t using std::lower_bound
    auto it = std::lower_bound(t_data.begin(), t_data.end(), t);
    size_t idx = std::distance(t_data.begin(), it) - 1;

    // Perform linear interpolation
    double t1 = t_data[idx], t2 = t_data[idx + 1];
    double k1 = k_data[idx], k2 = k_data[idx + 1];
    return k1 + (k2 - k1) * (t - t1) / (t2 - t1);
}

#endif // K_INTERPOLATION_H
