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

#ifndef ARGON2_OPENCL_GLOBALCONTEXT_H
#define ARGON2_OPENCL_GLOBALCONTEXT_H

#include "device.h"

#include <string>
#include <vector>

namespace argon2 {
namespace opencl {

class GlobalContext
{
private:
    std::vector<Device> devices;

public:
    const std::vector<Device> &getAllDevices() const { return devices; }

    GlobalContext();
};

} // namespace opencl
} // namespace argon2

#endif // ARGON2_OPENCL_GLOBALCONTEXT_H
