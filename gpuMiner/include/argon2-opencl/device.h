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

#ifndef ARGON2_OPENCL_DEVICE_H
#define ARGON2_OPENCL_DEVICE_H

#include "opencl.h"

namespace argon2 {
namespace opencl {

class Device
{
private:
    cl::Device device;

public:
    std::string getName() const;
    std::string getInfo() const;

    const cl::Device &getCLDevice() const { return device; }

    /**
     * @brief Empty constructor.
     * NOTE: Calling methods other than the destructor on an instance initialized
     * with empty constructor results in undefined behavior.
     */
    Device() { }

    Device(const cl::Device &device)
        : device(device)
    {
    }

    Device(const Device &) = default;
    Device(Device &&) = default;

    Device &operator=(const Device &) = default;
};

} // namespace opencl
} // namespace argon2

#endif // ARGON2_OPENCL_DEVICE_H
