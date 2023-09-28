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

#include <cuda_runtime.h>

#include "device.h"
#include "cudaexception.h"

namespace argon2 {
namespace cuda {

std::string Device::getName() const
{
    cudaDeviceProp prop;
    CudaException::check(cudaGetDeviceProperties(&prop, deviceIndex));
    return "CUDA Device '" + std::string(prop.name) + "'";
}

std::string Device::getInfo() const
{
    /* FIXME: show some more stuff here: */
    cudaDeviceProp prop;
    CudaException::check(cudaGetDeviceProperties(&prop, deviceIndex));
    return "CUDA Device '" + std::string(prop.name) + "'";
}

} // namespace cuda
} // namespace argon2
