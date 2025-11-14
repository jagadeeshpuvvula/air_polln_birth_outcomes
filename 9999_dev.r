
# %%
#https://publications.aap.org/pediatrics/article/125/2/e214/72106/New-Intrauterine-Growth-Curves-Based-on-United?autologincheck=redirected
olsen_ref <- tribble(
  ~sex, ~ga, ~mean_bw, ~sd_bw, ~p3_bw, ~p10_bw, ~p25_bw, ~p50_bw, ~p75_bw, ~p90_bw, ~p97_bw,
  
  # Females
  "F", 23, 587, 80, NA, 477, 528, 584, 639, 687, NA,
  "F", 24, 649, 89, 464, 524, 585, 651, 715, 772, 828,
  "F", 25, 738, 121, 511, 584, 657, 737, 816, 885, 953,
  "F", 26, 822, 143, 558, 645, 732, 827, 921, 1004, 1085,
  "F", 27, 934, 168, 615, 719, 822, 936, 1047, 1147, 1244,
  "F", 28, 1058, 203, 686, 807, 928, 1061, 1193, 1310, 1425,
  "F", 29, 1199, 226, 778, 915, 1052, 1204, 1354, 1489, 1621,
  "F", 30, 1376, 246, 902, 1052, 1204, 1373, 1542, 1693, 1842,
  "F", 31, 1548, 271, 1033, 1196, 1361, 1546, 1731, 1897, 2062,
  "F", 32, 1730, 300, 1177, 1352, 1530, 1731, 1933, 2116, 2297,
  "F", 33, 1960, 328, 1356, 1545, 1738, 1956, 2178, 2379, 2580,
  "F", 34, 2194, 357, 1523, 1730, 1944, 2187, 2434, 2661, 2888,
  "F", 35, 2420, 440, 1626, 1869, 2123, 2413, 2711, 2985, 3261,
  "F", 36, 2675, 514, 1745, 2028, 2324, 2664, 3015, 3339, 3667,
  "F", 37, 2946, 551, 1958, 2260, 2575, 2937, 3308, 3651, 3997,
  "F", 38, 3184, 512, 2235, 2526, 2829, 3173, 3525, 3847, 4172,
  "F", 39, 3342, 489, 2445, 2724, 3012, 3338, 3670, 3973, 4276,
  "F", 40, 3461, 465, 2581, 2855, 3136, 3454, 3776, 4070, 4363,
  "F", 41, 3546, 477, 2660, 2933, 3214, 3530, 3851, 4142, 4433,
  
  # Males
  "M", 23, 622, 74, NA, 509, 563, 621, 677, 727, NA,
  "M", 24, 689, 96, 497, 561, 623, 690, 756, 813, 869,
  "M", 25, 777, 116, 550, 626, 700, 780, 857, 926, 992,
  "M", 26, 888, 145, 613, 704, 794, 890, 983, 1065, 1145,
  "M", 27, 1001, 170, 680, 789, 895, 1009, 1120, 1218, 1312,
  "M", 28, 1138, 203, 758, 884, 1007, 1141, 1271, 1385, 1496,
  "M", 29, 1277, 218, 845, 988, 1128, 1280, 1429, 1560, 1688,
  "M", 30, 1435, 261, 955, 1114, 1272, 1443, 1612, 1761, 1906,
  "M", 31, 1633, 275, 1093, 1267, 1441, 1631, 1818, 1984, 2147,
  "M", 32, 1823, 306, 1246, 1433, 1622, 1829, 2034, 2218, 2398,
  "M", 33, 2058, 341, 1422, 1625, 1830, 2057, 2284, 2488, 2688,
  "M", 34, 2288, 364, 1589, 1810, 2035, 2285, 2536, 2763, 2987,
  "M", 35, 2529, 433, 1728, 1980, 2238, 2527, 2819, 3084, 3348,
  "M", 36, 2798, 498, 1886, 2170, 2462, 2792, 3127, 3432, 3737,
  "M", 37, 3058, 518, 2103, 2401, 2708, 3056, 3411, 3736, 4060,
  "M", 38, 3319, 527, 2356, 2652, 2959, 3306, 3661, 3986, 4312,
  "M", 39, 3476, 498, 2545, 2833, 3131, 3469, 3813, 4129, 4446,
  "M", 40, 3582, 493, 2666, 2950, 3245, 3579, 3919, 4232, 4545,
  "M", 41, 3691, 518, 2755, 3039, 3333, 3666, 4007, 4319, 4633
)



# %%
#' Fit DLNM model with separate exposure and outcome datasets
#'
#' @param exposure_data Dataframe containing exposure variables with common prefix
#' @param outcome_data Dataframe containing outcome and covariates
#' @param id_var Character string of ID variable name to merge datasets
#' @param exposure_prefix Character string prefix for exposure columns (e.g., "bc", "pm")
#' @param outcome_var Character string name of outcome variable
#' @param covariates Character vector of covariate names
#' @param var_df Degrees of freedom for exposure-response (default = 3)
#' @param var_degree Degree for B-spline on exposure (default = 2)
#' @param lag_df Degrees of freedom for lag structure (default = 3)
#' @param family Family for glm (default = binomial())
#' @param center_value How to center predictions: "median", "mean", or numeric value (default = "median")
#' @param n_pred_points Number of points for prediction grid (default = 50)
#'
#' @return List containing model, crosspred object, data used, and metadata
fit_dlnm <- function(exposure_data,
                     outcome_data,
                     id_var,
                     exposure_prefix,
                     outcome_var,
                     covariates,
                     var_df = 3,
                     var_degree = 2,
                     lag_df = 3,
                     family = binomial(),
                     center_value = "median",
                     n_pred_points = 50) {
  
  # Load required libraries
  require(tidyverse, quietly = TRUE)
  require(dlnm, quietly = TRUE)
  
  # Step 1: Merge datasets by ID
  merged_data <- outcome_data %>%
    inner_join(exposure_data, by = id_var)
  
  cat("After merging by", id_var, ":", nrow(merged_data), "observations\n")
  
  # Step 2: Extract exposure variables
  exposure_vars <- names(merged_data) %>% str_subset(paste0("^", exposure_prefix))
  
  if(length(exposure_vars) == 0) {
    stop("No exposure variables found with prefix '", exposure_prefix, "'")
  }
  
  cat("Found", length(exposure_vars), "exposure variables with prefix '", exposure_prefix, "'\n")
  
  # Step 3: Create exposure matrix
  Q_matrix <- as.matrix(merged_data[, exposure_vars])
  colnames(Q_matrix) <- paste0("lag", 0:(ncol(Q_matrix)-1))
  
  # Step 4: Handle missing values
  # Check for missing in exposure, outcome, and covariates
  all_vars <- c(exposure_vars, outcome_var, covariates)
  complete_rows <- complete.cases(merged_data[, all_vars])
  
  n_missing <- sum(!complete_rows)
  if(n_missing > 0) {
    cat("Dropping", n_missing, "observations with missing values\n")
    Q_matrix <- Q_matrix[complete_rows, ]
    merged_data <- merged_data[complete_rows, ]
  }
  
  cat("\n**Final sample size:", nrow(merged_data), "observations**\n\n")
  
  # Step 5: Create cross-basis matrix
  cb <- crossbasis(Q_matrix, 
                   lag = c(0, ncol(Q_matrix)-1),
                   argvar = list(fun = "bs", degree = var_degree, df = var_df),
                   arglag = list(fun = "ns", df = lag_df, intercept = FALSE))
  
  # Step 6: Build formula
  covariate_formula <- paste(covariates, collapse = " + ")
  formula_str <- paste(outcome_var, "~ cb +", covariate_formula)
  model_formula <- as.formula(formula_str)
  
  cat("Model formula:", formula_str, "\n\n")
  
  # Step 7: Fit model
  model <- glm(model_formula, family = family, data = merged_data)
  
  # Step 8: Create predictions
  # Determine centering value
  if(is.numeric(center_value)) {
    cen_value <- center_value
  } else if(center_value == "median") {
    cen_value <- median(Q_matrix, na.rm = TRUE)
  } else if(center_value == "mean") {
    cen_value <- mean(Q_matrix, na.rm = TRUE)
  } else {
    stop("center_value must be 'median', 'mean', or a numeric value")
  }
  
  temp_range <- range(Q_matrix, na.rm = TRUE)
  temp_values <- seq(temp_range[1], temp_range[2], length.out = n_pred_points)
  
  pred <- crosspred(cb, model, 
                    cen = cen_value,
                    at = temp_values)
  
  # Return results
  results <- list(
    model = model,
    pred = pred,
    data = merged_data,
    Q_matrix = Q_matrix,
    crossbasis = cb,
    metadata = list(
      n_obs = nrow(merged_data),
      n_exposure_lags = ncol(Q_matrix),
      exposure_prefix = exposure_prefix,
      outcome_var = outcome_var,
      covariates = covariates,
      center_value = cen_value,
      formula = formula_str
    )
  )
  
  cat("Model fitting complete!\n")
  cat("Centered at:", cen_value, "\n")
  
  return(results)
}

# Example usage:
# results <- fit_dlnm(
#   exposure_data = bc_data,
#   outcome_data = outcome_data,
#   id_var = "id",
#   exposure_prefix = "bc",
#   outcome_var = "ptb",
#   covariates = c("mothage", "mothrace")
# )
# 




# %%
#' Extract lag-specific effects at a target exposure level
#'
#' @param results Output from fit_dlnm()
#' @param target_percentile Percentile of exposure (e.g., 0.75 for 75th percentile)
#' @param target_value Specific exposure value (overrides target_percentile if provided)
#' @param exposure_name Label for the exposure (e.g., "Tmax", "PM2.5", "Black Carbon")
#' @param lag_multiplier Multiplier for lag values (default = 1)
#' @param lag_offset Offset to add to lag values (default = 0)
#'
#' @return Dataframe with Lag, OR, OR_low, OR_high, and exposure columns
extract_lag_effects <- function(results,
                                target_percentile = 0.75,
                                target_value = NULL,
                                exposure_name = NULL,
                                lag_multiplier = 1,
                                lag_offset = 0) {
  
  # Load required library
  require(tidyverse, quietly = TRUE)
  
  Q_matrix <- results$Q_matrix
  pred <- results$pred
  
  # Get temperature values used in prediction
  temp_values <- as.numeric(rownames(pred$matRRfit))
  
  # Determine target temperature
  if(!is.null(target_value)) {
    target_temp <- target_value
  } else {
    target_temp <- quantile(Q_matrix, target_percentile, na.rm = TRUE)
  }
  
  # Find closest temperature in prediction grid
  closest_temp_idx <- which.min(abs(temp_values - target_temp))
  closest_temp <- temp_values[closest_temp_idx]
  
  cat("Target exposure level:", target_temp, "\n")
  cat("Closest value in prediction grid:", closest_temp, "\n")
  cat("Reference (centered at):", results$metadata$center_value, "\n\n")
  
  # Extract lag-specific effects
  lag_values <- 0:(ncol(Q_matrix) - 1)
  lag_effects <- pred$matRRfit[closest_temp_idx, ]
  lag_lower <- pred$matRRlow[closest_temp_idx, ]
  lag_upper <- pred$matRRhigh[closest_temp_idx, ]
  
  # Create data frame
  lag_df <- data.frame(
    Lag = lag_values * lag_multiplier + lag_offset,
    OR = lag_effects,
    OR_low = lag_lower,
    OR_high = lag_upper
  )
  
  # Add exposure name if provided
  if(!is.null(exposure_name)) {
    lag_df <- lag_df %>% mutate(exposure = exposure_name)
  } else {
    lag_df <- lag_df %>% mutate(exposure = results$metadata$exposure_prefix)
  }
  
  return(lag_df)
}
