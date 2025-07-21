use numpy::{PyReadonlyArray2}; //from pyo3 package, converts Python numpy arrays in Rust data type (2d numpy arrays of f64) 
use pyo3::prelude::*; //for writing Python extensions in Rust
use pyo3::types::PyModule; //interaction with Python modules, wrapping Rust functions as a Python callable
use ndarray::{s, Array1, Array2}; //arrays in Rust, 1d and 2d

#[pyfunction] //py03 attribute macro, marks this as a function that can be called from Python code
fn athrust(py_momenta: PyReadonlyArray2<f64>) -> (f64, Vec<f64>) { //py_momenta: represents the input momenta as a 2D numpy array of f64 
    let momenta: Array2<f64> = py_momenta.as_array().to_owned(); //convert into a rust array, fix ownership
    let n = momenta.nrows(); //function to get the number of rows

    if n <= 2 {
        panic!("No thrust for 2 or fewer tracks."); //calculating thrust requires more than 2 tracks
    }

    let spatial = if n > 200 {
        eprintln!("Warning: {} tracks is too many. Using only the first 200.", n);
        momenta.slice(s![..200, ..]).to_owned() //take the first 200 tracks, create a new owned copy of the sliced array
    } else {
        momenta
    };
    

    let n = spatial.nrows();
    let mut vmax = 0.0; //initialize thrust value with zero
    let mut thrust_axis = Array1::<f64>::zeros(3); //initialize thrust axis with zeros

    for i in 0..(n - 1) {
        for j in (i + 1)..n { //nested loop, go through all pairs and calculate the cross product
            let vi = spatial.row(i); //get the ith and jth rows from the spatial array
            let vj = spatial.row(j);
            let vc = ndarray::arr1(&[ //cross product formula
                vi[1] * vj[2] - vi[2] * vj[1],
                vi[2] * vj[0] - vi[0] * vj[2],
                vi[0] * vj[1] - vi[1] * vj[0],
            ]);

            if vc.dot(&vc).sqrt() < 1e-15 { //check whether the vectors were parallel
                continue;
            }

            let vl = spatial.outer_iter().fold(Array1::<f64>::zeros(3), |acc, p| {
                if p.dot(&vc) >= 0.0 {
                    acc + &p
                } else {
                    acc - &p
                }
            }); //project all momenta on the direction of the cross-product. add those in the same direction, subtract those in the opposite direction

            for &sign_i in &[1.0, -1.0] { //loop to check the different sign combinations for i and j
                for &sign_j in &[1.0, -1.0] {
                    let vnew = &vl + &(sign_i * &spatial.row(i)) + &(sign_j * &spatial.row(j));
                    let vnorm = vnew.dot(&vnew);
                    if vnorm > vmax {
                        vmax = vnorm;
                        thrust_axis.assign(&vnew);
                    }
                }
            }
        }
    }

    for _ in 0..4 {
        let projected = spatial.outer_iter().map(|p| {
            if p.dot(&thrust_axis) >= 0.0 {
                p.to_owned()
            } else {
                -p.to_owned() //choose the right sign
            }
        });

        let vnew = projected.reduce(|a, b| a + b).unwrap(); 
        let vnorm = vnew.dot(&vnew);
        if (vnorm - vmax).abs() < 1e-12 {
            break;
        }
        vmax = vnorm;
        thrust_axis.assign(&vnew);
    }

    let total_momentum: f64 = spatial.outer_iter() //total momentum = summing the magnitudes of all momentum vectors
        .map(|p| p.dot(&p).sqrt())// thrust is ratio of square root of vmax to the total momentum
        .sum();

    let thrust = if total_momentum > 0.0 {
        vmax.sqrt() / total_momentum
    } else {
        0.0
    };

    let norm = thrust_axis.dot(&thrust_axis).sqrt(); //normalize the thrust axis to unit length
    if norm > 1e-15 {
        thrust_axis.mapv_inplace(|x| x / norm);
    }

    (thrust, thrust_axis.to_vec()) // return the result
}

#[pymodule] //defines the actual Python module that gets compiled from your Rust code and can be imported from Python using import athrust_accel
fn athrust_accel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(athrust, m)?)?;
    Ok(())
}