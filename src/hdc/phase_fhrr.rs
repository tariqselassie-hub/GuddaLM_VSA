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
use ndarray::{Array1, Zip};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CartesianFhrrVector {
    pub re: Array1<f32>,  // real components
    pub im: Array1<f32>,  // imaginary components
}

impl CartesianFhrrVector {
    pub fn new(re: Array1<f32>, im: Array1<f32>) -> Self {
        assert_eq!(re.len(), im.len());
        Self { re, im }
    }

    pub fn dim(&self) -> usize {
        self.re.len()
    }

    /// Create a zero vector of given dimension
    pub fn zero(dim: usize) -> Self {
        Self::new(Array1::zeros(dim), Array1::zeros(dim))
    }

    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let re: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let im: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self::new(Array1::from_vec(re), Array1::from_vec(im))
    }

    pub fn random_unit(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let re: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let im: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let v = Self::new(Array1::from_vec(re), Array1::from_vec(im));
        v.normalize()
    }

    pub fn normalize(&self) -> Self {
        let norm_sq = (&self.re * &self.re + &self.im * &self.im).mapv(|x| x.sqrt());
        let inv_norm = norm_sq.mapv(|x| if x > 0.0 { 1.0 / x } else { 0.0 });
        Self::new(&self.re * &inv_norm, &self.im * &inv_norm)
    }

    // Binding: (a+bi)(c+di) = (ac - bd) + (ad + bc)i
    pub fn bind(&self, other: &Self) -> Self {
        Self::new(
            &self.re * &other.re - &self.im * &other.im,
            &self.re * &other.im + &self.im * &other.re,
        )
    }

    // Unbinding: a.unbind(b) = a.bind(b.inverse()) = a.bind(b.conjugate())
    pub fn unbind(&self, other: &Self) -> Self {
        let other_conj = Self::new(other.re.clone(), -&other.im);
        self.bind(&other_conj)
    }

    // Bundling: (a+bi) + (c+di) = (a+c) + (b+d)i
    pub fn bundle(&self, other: &Self) -> Self {
        Self::new(&self.re + &other.re, &self.im + &other.im)
    }

    pub fn bundle_accumulate(&mut self, other: &Self) {
        self.re += &other.re;
        self.im += &other.im;
    }

    // Similarity: cos(theta_a - theta_b) = (a·c + b·d) / (|a+bi||c+di|)
    pub fn similarity(&self, other: &Self) -> f32 {
        let dot = (&self.re * &other.re + &self.im * &other.im).sum();
        let norm_a_sq = (&self.re * &self.re + &self.im * &self.im).sum();
        let norm_b_sq = (&other.re * &other.re + &other.im * &other.im).sum();
        if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
            0.0
        } else {
            dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())
        }
    }

    /// Cosine similarity alias for compatibility
    pub fn cosine_similarity(&self, other: &Self) -> f64 {
        self.similarity(other) as f64
    }

    pub fn permute(&self, shift: isize) -> Self {
        let len = self.re.len();
        let shift_offset = ((shift % len as isize) + len as isize) as usize % len;
        if shift_offset == 0 {
            return self.clone();
        }
        // Right rotation by shift_offset
        let re_part1 = self.re.slice(ndarray::s![len - shift_offset..]).to_owned();
        let re_part2 = self.re.slice(ndarray::s![..len - shift_offset]).to_owned();
        let im_part1 = self.im.slice(ndarray::s![len - shift_offset..]).to_owned();
        let im_part2 = self.im.slice(ndarray::s![..len - shift_offset]).to_owned();
        Self::new(re_part1 + re_part2, im_part1 + im_part2)
    }
}

// Keep old PhaseFhrrVector for backward compatibility
#[derive(Clone)]
pub struct PhaseFhrrVector {
    pub phases: Array1<f32>,
}

impl PhaseFhrrVector {
    pub fn new(phases: Array1<f32>) -> Self {
        let bounded = phases.mapv(|theta| theta.rem_euclid(TAU));
        Self { phases: bounded }
    }

    pub fn random(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let phases: Vec<f32> = (0..dim).map(|_| rng.gen_range(0.0..TAU)).collect();
        Self { phases: Array1::from_vec(phases) }
    }

    pub fn bind(&self, other: &Self) -> Self {
        let bound_phases = &self.phases + &other.phases;
        Self::new(bound_phases)
    }

    pub fn inverse(&self) -> Self {
        let inv_phases = self.phases.mapv(|theta| TAU - theta);
        Self::new(inv_phases)
    }

    pub fn bundle(&self, other: &Self) -> Self {
        let mut result = Array1::zeros(self.phases.raw_dim());
        Zip::from(&mut result)
            .and(&self.phases)
            .and(&other.phases)
            .for_each(|res, &theta_a, &theta_b| {
                let re = theta_a.cos() + theta_b.cos();
                let im = theta_a.sin() + theta_b.sin();
                *res = im.atan2(re).rem_euclid(TAU);
            });
        Self { phases: result }
    }

    pub fn bundle_accumulate(&mut self, other: &Self) {
        Zip::from(&mut self.phases)
            .and(&other.phases)
            .for_each(|res, &theta_b| {
                let re = res.cos() + theta_b.cos();
                let im = res.sin() + theta_b.sin();
                *res = im.atan2(re).rem_euclid(TAU);
            });
    }

    pub fn similarity(&self, other: &Self) -> f32 {
        let dimension = self.phases.len() as f32;
        let mut cos_sum = 0.0;
        Zip::from(&self.phases)
            .and(&other.phases)
            .for_each(|&theta_a, &theta_b| {
                cos_sum += (theta_a - theta_b).cos();
            });
        cos_sum / dimension
    }

    pub fn permute(&self, shift: isize) -> Self {
        let len = self.phases.len();
        let mut permuted = Array1::zeros(self.phases.raw_dim());
        let shift_offset = ((shift % len as isize) + len as isize) as usize % len;
        if shift_offset == 0 {
            return Self { phases: self.phases.clone() };
        }
        let (left, right) = self.phases.view().split_at(ndarray::Axis(0), len - shift_offset);
        permuted.slice_mut(ndarray::s![..shift_offset]).assign(&right);
        permuted.slice_mut(ndarray::s![shift_offset..]).assign(&left);
        Self { phases: permuted }
    }
}
