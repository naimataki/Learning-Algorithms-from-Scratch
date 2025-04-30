# Understanding Learned vs. True Parameters with Normalization in Linear Regression

## Observation

You might notice when running the linear regression code that the learned `theta` parameters (e.g., `[13.877, 6.011]`) don't directly match the `bias` (e.g., 4) and `theta_true` (e.g., 2) values used to generate the synthetic data.

This is **expected behaviour** and not an error in the learning algorithm itself.

## The Core Reason: Feature Normalization

The key reason for this difference is **feature normalization**.

* The **true parameters** (`bias`, `theta_true`) define the relationship between the target `y` and the **original, unnormalized** features `X`.
* The **learned parameters** (`theta`) define the relationship between the target `y` and the **normalized** features `X_norm`.

Since `X` and `X_norm` are on different scales, the parameters describing their relationship with `y` will also be different.

## Mathematical Derivation

Let's walk through the relationship:

1.  **Original Data Model:**
    The synthetic data is generated based on the original features `X`:
    `y = bias + X * theta_true + noise`
    (In our example: `y = 4 + X * 2 + noise`)

2.  **Normalization Step:**
    Before training, the features `X` are normalized:
    `X_norm = (X - \mu_X) / \sigma_X`
    Where $\mu_X$ is the mean and $\sigma_X$ is the standard deviation of the *original* feature data `X`.

3.  **Model Trained on Normalized Data:**
    The linear regression model learns parameters $\theta_0$ (learned bias/intercept) and $\theta_1$ (learned feature coefficient) based on the normalized features `X_norm` (after adding a column of ones for the intercept term):
    `y ≈ \theta_0 * 1 + X_norm * \theta_1`

4.  **Connecting the Models:**
    Substitute the definition of `X_norm` into the learned model equation:
    `y ≈ \theta_0 + ((X - \mu_X) / \sigma_X) * \theta_1`

    Now, rearrange this equation to look like the original model form (y ≈ intercept + coefficient * X):
    `y ≈ \theta_0 + (\theta_1 / \sigma_X) * X - (\mu_X * \theta_1) / \sigma_X`
    `y ≈ (\theta_0 - (\mu_X * \theta_1) / \sigma_X) + (\theta_1 / \sigma_X) * X`

5.  **Comparing Coefficients:**
    By comparing this rearranged equation to the original data model (`y = bias + X * theta_true`), we can see the relationship:

    * `bias` (Original Intercept) corresponds to `(\theta_0 - (\mu_X * \theta_1) / \sigma_X)` (Effective Intercept from Learned Parameters)
    * `theta_true` (Original Coefficient) corresponds to `(\theta_1 / \sigma_X)` (Effective Coefficient from Learned Parameters)

## How to Convert Learned Parameters Back

You can estimate the original `bias` and `theta_true` from the learned parameters $\theta_0$ and $\theta_1$ if you know the mean ($\mu_X$) and standard deviation ($\sigma_X$) used for normalization:

* **Estimated `theta_true`** $\approx \frac{\theta_1}{\sigma_X}$
* **Estimated `bias`** $\approx \theta_0 - \frac{\mu_X \cdot \theta_1}{\sigma_X}$

Remember:
* $\theta_0$ is the first element of the learned `theta` vector.
* $\theta_1$ is the second element of the learned `theta` vector.
* $\mu_X$ and $\sigma_X$ are the mean and standard deviation **of the original `X` data**, which should be returned by the `train_model` function alongside the learned `theta`.

## Example Calculation

Using the example values:
* Learned `theta`: `[\theta_0, \theta_1] = [13.877, 6.011]`
* True Parameters: `bias = 4`, `theta_true = 2`

We need the *actual* mean ($\mu_X$) and std ($\sigma_X$) from the normalization step in the code. If `X` was generated using `np.random.rand(100, 1) * 10`, we can *estimate*:
* $\mu_X \approx 5.0$
* $\sigma_X \approx \sqrt{(10-0)^2 / 12} \approx 2.887$

Plugging these estimates into the conversion formulas:

* Estimated `theta_true` $\approx \frac{6.011}{2.887} \approx 2.08$ (Close to the true value of 2!)
* Estimated `bias` $\approx 13.877 - \frac{5.0 \times 6.011}{2.887} \approx 13.877 - (5.0 \times 2.08) \approx 13.877 - 10.4 \approx 3.48$ (Reasonably close to the true value of 4!)

**Note:** Use the *precise* `mean` and `std` values returned by your `train_model` function for the most accurate conversion. The closeness of the converted values to the true parameters depends on noise, model convergence, and data variability.

## Other Influencing Factors

Besides normalization, other factors can cause the converted parameters to not perfectly match the true ones:

* **Noise:** Random noise added during data generation means the model fits a slightly distorted version of the true relationship.
* **Convergence:** The gradient descent might not have fully converged. Check the loss curve; consider running for more `epochs` or adjusting the `alpha` (learning rate).
* **Data Sample Size:** With a limited number of samples (e.g., 100), the specific random values in `X` and the noise can significantly influence the learned parameters.

## Summary

The learned `theta` parameters are correct for the **normalized scale** the model was trained on. They differ from the true parameters because the true parameters apply to the **original data scale**. You can mathematically convert the learned parameters back to the original scale using the mean and standard deviation from the normalization step, which helps verify that the model learned the underlying relationship correctly, accounting for the effects of normalization and noise.