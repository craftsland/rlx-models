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

//! The log-mel frontend, shared with the other NeMo models in
//! [`rlx_mcu::mel`]; this model supplies its stored window and banded
//! filterbank.

pub use rlx_mcu::mel::{Frontend, MAX_FFT, MelSource};

use crate::model::Model;

impl MelSource for Model<'_> {
    fn hop(&self) -> usize {
        self.config().hop
    }
    fn n_mels(&self) -> usize {
        self.config().n_mels
    }
    fn window(&self) -> &[f32] {
        Model::window(self)
    }
    fn mel_band(&self, m: usize) -> (usize, &[f32]) {
        Model::mel_band(self, m)
    }
}
