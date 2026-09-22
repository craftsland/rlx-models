// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! One module per architecture family, each a direct transliteration of the
//! reference PyTorch module it ports.

pub mod compact;
pub mod dat;
pub mod dysample;
pub mod esrgan;
pub mod mambair;
pub mod omnisr;
pub mod plksr;
pub mod realcugan;
pub mod safmn;
pub mod span;
pub mod swin;
