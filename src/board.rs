// Copyright (C) 2025 guddalm_vsa contributors.
// SPDX-License-Identifier: AGPL-3.0
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
pub mod primitives;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum File {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
    G = 6,
    H = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rank {
    R1 = 0,
    R2 = 1,
    R3 = 2,
    R4 = 3,
    R5 = 4,
    R6 = 5,
    R7 = 6,
    R8 = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

#[inline(always)]
pub fn make_square(file: File, rank: Rank) -> Square {
    Square(rank as u8 * 8 + file as u8)
}

#[inline(always)]
pub fn file_from_index(index: u8) -> File {
    match (index % 8) as u8 {
        0 => File::A,
        1 => File::B,
        2 => File::C,
        3 => File::D,
        4 => File::E,
        5 => File::F,
        6 => File::G,
        _ => File::H,
    }
}

#[inline(always)]
pub fn rank_from_index(index: u8) -> Rank {
    match (index / 8) as u8 {
        0 => Rank::R1,
        1 => Rank::R2,
        2 => Rank::R3,
        3 => Rank::R4,
        4 => Rank::R5,
        5 => Rank::R6,
        6 => Rank::R7,
        _ => Rank::R8,
    }
}
