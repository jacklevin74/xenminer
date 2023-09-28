/*
 * Copyright (C) 2015, Ondrej Mosnacek <omosnacek@gmail.com>
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation: either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "device.h"

#include <sstream>
#include <unordered_map>
#include <stdexcept>

namespace argon2 {
namespace opencl {

std::string Device::getName() const
{
    return "OpenCL Device '"
            + device.getInfo<CL_DEVICE_NAME>()
            + "' (" + device.getInfo<CL_DEVICE_VENDOR>() + ")";
}

template<class T>
static std::ostream &printBitfield(std::ostream &out, T value,
                                   const std::vector<std::pair<T, std::string>> &lookup)
{
    bool first = true;
    for (auto &entry : lookup) {
        if (value & entry.first) {
            if (!first) {
                out << " | ";
            }
            first = false;
            out << entry.second;
        }
    }
    return out;
}

template <class T>
static std::ostream &printEnum(std::ostream &out, T value,
                               const std::unordered_map<T, std::string> &lookup)
{
    try {
        return out << lookup.at(value);
    } catch (const std::out_of_range &) {
        return out << "<invalid>";
    }
}

template<class T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec)
{
    out << "[";
    bool first = true;
    for (T value : vec) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << value;
    }
    return out << "]";
}

std::string Device::getInfo() const
{
    std::ostringstream out;
    out << "OpenCL Device '" << device.getInfo<CL_DEVICE_NAME>() << "':" << std::endl;
    out << "  Type: ";
    printBitfield(out, device.getInfo<CL_DEVICE_TYPE>(), {
                      { CL_DEVICE_TYPE_CPU, "CPU" },
                      { CL_DEVICE_TYPE_GPU, "GPU" },
                      { CL_DEVICE_TYPE_ACCELERATOR, "Accelerator" },
                      { CL_DEVICE_TYPE_DEFAULT, "Default" },
                  }) << std::endl;
    out << "  Available: "
        << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;
    out << "  Compiler available: "
        << device.getInfo<CL_DEVICE_COMPILER_AVAILABLE>() << std::endl;
    out << std::endl;

    out << "  Version: "
        << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
    out << "  OpenCL C Version: "
        << device.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
    out << "  Extensions: "
        << device.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
    out << std::endl;

    out << "  Vendor: "
        << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
    out << "  Vendor ID: "
        << device.getInfo<CL_DEVICE_VENDOR_ID>() << std::endl;
    out << std::endl;

    cl::Platform platform(device.getInfo<CL_DEVICE_PLATFORM>());
    out << "  Platform name: "
        << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    out << "  Platform vendor: "
        << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
    out << "  Platform version: "
        << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
    out << "  Platform extensions: "
        << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << std::endl;
    out << std::endl;

    out << "  Driver version: "
        << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
    out << "  Little-endian: "
        << device.getInfo<CL_DEVICE_ENDIAN_LITTLE>() << std::endl;
    out << std::endl;

    out << "  Max compute units: "
        << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
    out << "  Max work-item dimensions: "
        << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
    out << "  Max work-item sizes: "
        << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << std::endl;
    out << std::endl;

    out << "  Max clock frequency: "
        << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
    out << std::endl;

    out << "  Address bits: "
        << device.getInfo<CL_DEVICE_ADDRESS_BITS>() << std::endl;
    out << "  Max memory allocation size: "
        << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << " bytes" << std::endl;
    out << "  Max parameter size: "
        << device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>() << " bytes" << std::endl;
    out << "  Memory base address alignment: "
        << device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>() << " bits" << std::endl;
    out << "  Min data type alignment: "
        << device.getInfo<CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE>() << " bytes" << std::endl;
    out << std::endl;

    out << "  Unified memory: "
        << device.getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>() << std::endl;
    out << "  Global memory cache type: ";
    printEnum(out, device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE>(), {
                  { CL_NONE, "None" },
                  { CL_READ_ONLY_CACHE, "Read-only" },
                  { CL_READ_WRITE_CACHE, "Read-write" }
              }) << std::endl;
    out << "  Global memory cacheline size: "
        << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE>() << " bytes" << std::endl;
    out << "  Global memory cache size: "
        << device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>() << " bytes" << std::endl;
    out << "  Global memory size: "
        << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << " bytes" << std::endl;
    out << std::endl;

    out << "  Max constant buffer size: "
        << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << " bytes" << std::endl;
    out << "  Max constant arguments: "
        << device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>() << std::endl;
    out << std::endl;

    out << "  Local memory type: ";
    printEnum(out, device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>(), {
                  { CL_LOCAL, "Dedicated" },
                  { CL_GLOBAL, "Global" },
              }) << std::endl;
    out << "  Local memory size: "
        << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " bytes" << std::endl;
    out << std::endl;

    out << "  Preferred vector width (char): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR>()
        << ")" << std::endl;
    out << "  Preferred vector width (short): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT>()
        << ")" << std::endl;
    out << "  Preferred vector width (int): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_INT>()
        << ")" << std::endl;
    out << "  Preferred vector width (long): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG>()
        << ")" << std::endl;
    out << "  Preferred vector width (float): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT>()
        << ")" << std::endl;
    out << "  Preferred vector width (double): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE>()
        << ")" << std::endl;
    out << "  Preferred vector width (half): "
        << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF>()
        << " (native: "
        << device.getInfo<CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF>()
        << ")" << std::endl;
    out << std::endl;

    out << "  Error correction supported: "
        << device.getInfo<CL_DEVICE_ERROR_CORRECTION_SUPPORT>() << std::endl;
    out << "  Profiling timer resolution: "
        << device.getInfo<CL_DEVICE_PROFILING_TIMER_RESOLUTION>() << " ns" << std::endl;
    out << std::endl;

    out << "  Execution capabilites: ";
    printBitfield(out, device.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>(), {
                      { CL_EXEC_KERNEL, "OpenCL kernels" },
                      { CL_EXEC_NATIVE_KERNEL, "Native kernels" },
                  }) << std::endl;
    out << "  Command queue properties: ";
    printBitfield(out, device.getInfo<CL_DEVICE_QUEUE_PROPERTIES>(), {
                      { CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "Out-of-order execution" },
                      { CL_QUEUE_PROFILING_ENABLE, "Profiling" },
                  }) << std::endl;
    return out.str();
}

} // namespace opencl
} // namespace argon2
