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

#include "globalcontext.h"

#include <iostream>

namespace argon2 {
namespace opencl {

GlobalContext::GlobalContext()
    : devices()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> clDevices;
    for (cl::Platform platform : platforms) {
        try {
            platform.getDevices(CL_DEVICE_TYPE_ALL, &clDevices);
            devices.insert(devices.end(), clDevices.begin(), clDevices.end());
        } catch (const cl::Error &err) {
            std::cerr << "WARNING: Unable to get devices for platform '"
                      << platform.getInfo<CL_PLATFORM_NAME>()
                      << "' - error " << err.err() << std::endl;
        }
    }
}

} // namespace opencl
} // namespace argon2
