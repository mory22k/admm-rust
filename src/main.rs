/*
Copyright 2025 mory22k

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use nalgebra as na;

fn soft_thresholding(x: &f64, threshold: &f64) -> f64 {
    // returns the soft thresholding of x with threshold
    if x > threshold {
        return x - threshold;
    } else if x < &-threshold {
        return x + threshold;
    } else {
        return 0.0;
    }
}

fn soft_thresholding_vec(x_vec: &na::DVector<f64>, threshold_vec: &na::DVector<f64>) -> na::DVector<f64> {
    // returns the soft thresholding of each element of x_vec with the corresponding element of threshold_vec
    let mut result_vec: na::DVector<f64> = na::DVector::zeros(x_vec.len());
    let num_variables: usize = x_vec.len();
    for i in 0..num_variables {
        result_vec[i] = soft_thresholding(&x_vec[i], &threshold_vec[i]);
    }
    return result_vec;
}

fn admm(
    x_mat: na::DMatrix<f64>,
    y_vec: na::DVector<f64>,
    lam_vec: na::DVector<f64>,
    z_init_vec: na::DVector<f64>,
    u_init_vec: na::DVector<f64>,
    rho_vec: na::DVector<f64>,
    max_iter: usize,
    return_history: bool,
    show_progress: bool,
) -> (na::DVector<f64>, na::DVector<f64>, na::DVector<f64>, Vec<na::DVector<f64>>, Vec<na::DVector<f64>>, Vec<na::DVector<f64>>) {
    let (num_data_points, num_variables): (usize, usize) = x_mat.shape();

    assert!(y_vec.shape() == (num_data_points, 1));
    assert!(lam_vec.shape() == (num_variables, 1));
    assert!(z_init_vec.shape() == (num_variables, 1));
    assert!(u_init_vec.shape() == (num_variables, 1));
    assert!(rho_vec.shape() == (num_variables, 1));

    let rho_diag: na::DMatrix<f64> = na::DMatrix::from_diagonal(&rho_vec);
    let a_mat: na::DMatrix<f64> = x_mat.transpose() * &x_mat + &rho_diag;
    let a_mat_chol= a_mat.cholesky().unwrap();
    let threshold_vec: na::DVector<f64> = lam_vec.component_div(&rho_vec);

    let mut w_vec = na::DVector::zeros(num_variables);
    let mut z_vec = z_init_vec;
    let mut u_vec = u_init_vec;

    let mut w_vec_history: Vec<na::DVector<f64>> = Vec::new();
    let mut z_vec_history: Vec<na::DVector<f64>> = Vec::new();
    let mut u_vec_history: Vec<na::DVector<f64>> = Vec::new();

    for t in 0..max_iter {
        let b: na::DVector<f64> = &x_mat.transpose() * &y_vec + &rho_diag * (&z_vec - &u_vec);
        w_vec = a_mat_chol.solve( &b );

        let w_vec_plus_u_vec: na::DVector<f64> = &w_vec + &u_vec;
        z_vec = soft_thresholding_vec( &w_vec_plus_u_vec, &threshold_vec );
        u_vec = &u_vec + &w_vec - &z_vec;

        if return_history {
            w_vec_history.push(w_vec.clone());
            z_vec_history.push(z_vec.clone());
            u_vec_history.push(u_vec.clone());
        }

        if show_progress {
            let rmse: f64 = (&x_mat * &w_vec - &y_vec).norm() / f64::sqrt(num_variables as f64);
            println!("[{:>4} / {}] {}", t, max_iter, rmse);
        }
    }

    return (w_vec, z_vec, u_vec, w_vec_history, z_vec_history, u_vec_history);
}

fn main() {
    let num_variables: usize = 256;
    let num_data_points: usize = 128;
    let density: f64 = 0.5;
    let rho: f64 = 1.0;
    let lam: f64 = 1.0;
    let max_iter: usize = 100;

    let x_true_mat: na::DMatrix<f64> =
        na::DMatrix::new_random(num_data_points, num_variables) * 2.0
        - na::DMatrix::from_element(num_data_points, num_variables, 1.0);

    let w_true_vec_row: na::DVector<f64> =
        na::DVector::new_random(num_variables) * 2.0
        - na::DVector::from_element(num_variables, 1.0);
    let w_true_vec_mask: na::DVector<f64> =
        na::DVector::new_random(num_variables).map(|x:f64| if x < density { 1.0 } else { 0.0 });
    let w_true_vec: na::DVector<f64> = w_true_vec_row.component_mul(&w_true_vec_mask);

    let y_true_vec: na::DVector<f64> =
        &x_true_mat * &w_true_vec;

    let z_init_vec: na::DVector<f64> =
        na::DVector::from_element(num_variables, 0.0);
    let u_init_vec: na::DVector<f64> =
        na::DVector::from_element(num_variables, 0.0);
    let rho_vec: na::DVector<f64> =
        na::DVector::from_element(num_variables, rho);
    let lam_vec: na::DVector<f64> =
        na::DVector::from_element(num_variables, lam);

    let (_w_est_vec, _, _, _, _, _) = admm(
        x_true_mat,
        y_true_vec,
        lam_vec,
        z_init_vec,
        u_init_vec,
        rho_vec,
        max_iter,
        false,
        true
    );

    let mse: f64 = (&w_true_vec - &_w_est_vec).norm_squared() / num_variables as f64;
    println!("MSE of w_est vs. w_true: {}", mse);
}
