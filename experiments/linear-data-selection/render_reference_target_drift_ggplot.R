#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript render_reference_target_drift_ggplot.R <output_dir>")
}

output_dir <- normalizePath(args[[1]], mustWork = TRUE)
summary_path <- file.path(output_dir, "reference_target_regret_summary.csv")
config_path <- file.path(output_dir, "config.json")

summary_df <- read.csv(summary_path, stringsAsFactors = FALSE)
config <- fromJSON(config_path)
rho <- config$grid_config$dimred_config$rho

method_labels <- c(
  "safe-full" = "SAFE full",
  "safe-lowrank" = "SAFE low-rank",
  "safe-reference-only" = "SAFE reference-only",
  "safe-task-only" = "SAFE task-only",
  "safe-random-k" = "SAFE random-K",
  "safe-diagonal" = "SAFE diagonal",
  "baseline-less" = "LESS",
  "baseline-random" = "random",
  "baseline-prismatic" = "PRISM"
)

method_colors <- c(
  "safe-full" = "#e41a1c",
  "safe-lowrank" = "#377eb8",
  "safe-reference-only" = "#984ea3",
  "safe-task-only" = "#4daf4a",
  "safe-random-k" = "#ff7f00",
  "safe-diagonal" = "#a65628",
  "baseline-less" = "#17becf",
  "baseline-random" = "#8c564b",
  "baseline-prismatic" = "#bcbd22"
)

safe_methods <- c(
  "safe-full",
  "safe-lowrank",
  "safe-reference-only",
  "safe-task-only",
  "safe-random-k",
  "safe-diagonal"
)

setting_title <- function(prefix, label) {
  if (grepl("^powerlaw-q", label)) {
    exponent <- sub("^powerlaw-q", "", label)
    return(sprintf("%s powerlaw (q=%s)", prefix, exponent))
  }
  sprintf("%s %s", prefix, label)
}

add_offsets <- function(df) {
  methods <- unique(df$method)
  methods <- methods[methods %in% names(method_labels)]
  methods <- methods[order(match(methods, names(method_labels)))]
  if (length(methods) == 1) {
    offsets <- c(0.0)
  } else {
    offsets <- seq(-0.14, 0.14, length.out = length(methods))
  }
  offset_map <- setNames(offsets, methods)
  x_levels <- sort(unique(df$K_requested))
  x_map <- setNames(seq_along(x_levels), as.character(x_levels))
  df$x_base <- unname(x_map[as.character(df$K_requested)])
  df$x_plot <- df$x_base + unname(offset_map[df$method])
  df$drift_ymin <- pmax(1e-6, df$drift_mean - df$drift_sem)
  df$drift_ymax <- pmax(df$drift_mean + df$drift_sem, df$drift_ymin * 1.001)
  df$regret_ymin <- df$absolute_regret_mean - df$absolute_regret_sem
  df$regret_ymax <- df$absolute_regret_mean + df$absolute_regret_sem
  df
}

make_drift_plot <- function(df, row_order, col_order, methods, figure_title, output_file) {
  plot_df <- df[df$method %in% methods, , drop = FALSE]
  if (nrow(plot_df) == 0) {
    return(invisible(NULL))
  }

  plot_df$method <- factor(plot_df$method, levels = methods)
  plot_df$method_label <- factor(method_labels[as.character(plot_df$method)], levels = method_labels[methods])
  plot_df$reference_setting <- factor(plot_df$reference_setting, levels = row_order)
  plot_df$target_setting <- factor(plot_df$target_setting, levels = col_order)
  plot_df$reference_label <- factor(
    vapply(as.character(plot_df$reference_setting), function(x) setting_title("reference", x), character(1)),
    levels = vapply(row_order, function(x) setting_title("reference", x), character(1))
  )
  plot_df$target_label <- factor(
    vapply(as.character(plot_df$target_setting), function(x) setting_title("target", x), character(1)),
    levels = vapply(col_order, function(x) setting_title("target", x), character(1))
  )

  plot_df <- add_offsets(plot_df)
  x_levels <- sort(unique(plot_df$K_requested))
  x_breaks <- seq_along(x_levels)
  x_labels <- as.character(x_levels)
  color_values <- setNames(unname(method_colors[methods]), method_labels[methods])

  p <- ggplot(plot_df, aes(x = x_plot, y = drift_mean, color = method_label, group = method_label)) +
    geom_line(linewidth = 0.6, alpha = 0.9) +
    geom_errorbar(aes(ymin = drift_ymin, ymax = drift_ymax), width = 0.04, linewidth = 0.45, alpha = 0.85) +
    geom_point(size = 1.6, alpha = 0.95) +
    facet_grid(reference_label ~ target_label, scales = "free") +
    scale_color_manual(values = color_values, breaks = method_labels[methods], drop = FALSE) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels) +
    scale_y_log10() +
    labs(
      title = figure_title,
      x = "Requested rank K_R + K_T",
      y = "Mean reference drift (log)",
      color = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3),
      panel.grid.major.y = element_line(color = "grey80", linewidth = 0.35),
      legend.position = "bottom",
      legend.box = "horizontal",
      axis.text = element_text(size = 12, color = "black"),
      axis.title = element_text(size = 13, color = "black"),
      legend.text = element_text(size = 11),
      strip.text = element_text(size = 12, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
    )

  if (!is.null(rho) && !is.na(rho)) {
    p <- p + geom_hline(yintercept = rho, color = "#c62828", linetype = "dashed", linewidth = 0.5)
  }

  ggsave(
    filename = file.path(output_dir, output_file),
    plot = p,
    width = 5.1 * length(col_order),
    height = 4.0 * length(row_order),
    dpi = 180,
    bg = "white"
  )
}

make_regret_plot <- function(df, row_order, col_order, methods, figure_title, output_file) {
  plot_df <- df[df$method %in% methods, , drop = FALSE]
  if (nrow(plot_df) == 0) {
    return(invisible(NULL))
  }

  plot_df$method <- factor(plot_df$method, levels = methods)
  plot_df$method_label <- factor(method_labels[as.character(plot_df$method)], levels = method_labels[methods])
  plot_df$reference_setting <- factor(plot_df$reference_setting, levels = row_order)
  plot_df$target_setting <- factor(plot_df$target_setting, levels = col_order)
  plot_df$reference_label <- factor(
    vapply(as.character(plot_df$reference_setting), function(x) setting_title("reference", x), character(1)),
    levels = vapply(row_order, function(x) setting_title("reference", x), character(1))
  )
  plot_df$target_label <- factor(
    vapply(as.character(plot_df$target_setting), function(x) setting_title("target", x), character(1)),
    levels = vapply(col_order, function(x) setting_title("target", x), character(1))
  )

  plot_df <- add_offsets(plot_df)
  x_levels <- sort(unique(plot_df$K_requested))
  x_breaks <- seq_along(x_levels)
  x_labels <- as.character(x_levels)
  color_values <- setNames(unname(method_colors[methods]), method_labels[methods])

  p <- ggplot(plot_df, aes(x = x_plot, y = absolute_regret_mean, color = method_label, group = method_label)) +
    geom_line(linewidth = 0.6, alpha = 0.9) +
    geom_errorbar(aes(ymin = regret_ymin, ymax = regret_ymax), width = 0.04, linewidth = 0.45, alpha = 0.85) +
    geom_point(size = 1.6, alpha = 0.95) +
    facet_grid(reference_label ~ target_label, scales = "free") +
    scale_color_manual(values = color_values, breaks = method_labels[methods], drop = FALSE) +
    scale_x_continuous(breaks = x_breaks, labels = x_labels) +
    scale_y_continuous(trans = scales::pseudo_log_trans(base = 10, sigma = 0.05)) +
    labs(
      title = figure_title,
      x = "Requested rank K_R + K_T",
      y = "Absolute regret vs SAFE full (pseudo-log)",
      color = NULL
    ) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3),
      panel.grid.major.y = element_line(color = "grey80", linewidth = 0.35),
      legend.position = "bottom",
      legend.box = "horizontal",
      axis.text = element_text(size = 12, color = "black"),
      axis.title = element_text(size = 13, color = "black"),
      legend.text = element_text(size = 11),
      strip.text = element_text(size = 12, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
    ) +
    geom_hline(yintercept = 0.0, color = "#666666", linetype = "dashed", linewidth = 0.5)

  ggsave(
    filename = file.path(output_dir, output_file),
    plot = p,
    width = 5.1 * length(col_order),
    height = 4.0 * length(row_order),
    dpi = 180,
    bg = "white"
  )
}

family_reference <- unique(summary_df$reference_setting[!grepl("^powerlaw-q(0\\.5|0\\.75|1\\.5|2)$", summary_df$reference_setting)])
family_target <- unique(summary_df$target_setting[!grepl("^powerlaw-q(0\\.5|0\\.75|1\\.5|2)$", summary_df$target_setting)])

# Keep family order stable for the 3x3 grid.
family_reference <- family_reference[family_reference %in% c("geometric", "linear", "powerlaw-q1")]
family_target <- family_target[family_target %in% c("geometric", "linear", "powerlaw-q1")]

powerlaw_reference <- sort(unique(summary_df$reference_setting[grepl("^powerlaw-q", summary_df$reference_setting)]))
powerlaw_target <- sort(unique(summary_df$target_setting[grepl("^powerlaw-q", summary_df$target_setting)]))

make_drift_plot(
  summary_df[summary_df$reference_setting %in% family_reference & summary_df$target_setting %in% family_target, , drop = FALSE],
  row_order = family_reference,
  col_order = family_target,
  methods = names(method_labels),
  figure_title = "Reference drift: reference vs target decay families",
  output_file = "reference_drift_reference_vs_target_decay_all_methods.png"
)

make_drift_plot(
  summary_df[summary_df$reference_setting %in% family_reference & summary_df$target_setting %in% family_target, , drop = FALSE],
  row_order = family_reference,
  col_order = family_target,
  methods = safe_methods,
  figure_title = "Reference drift: reference vs target decay families (SAFE variants)",
  output_file = "reference_drift_reference_vs_target_decay_safe_methods.png"
)

make_regret_plot(
  summary_df[summary_df$reference_setting %in% family_reference & summary_df$target_setting %in% family_target, , drop = FALSE],
  row_order = family_reference,
  col_order = family_target,
  methods = names(method_labels),
  figure_title = "Absolute regret vs SAFE full: reference vs target decay families",
  output_file = "absolute_regret_reference_vs_target_decay_all_methods.png"
)

make_regret_plot(
  summary_df[summary_df$reference_setting %in% family_reference & summary_df$target_setting %in% family_target, , drop = FALSE],
  row_order = family_reference,
  col_order = family_target,
  methods = safe_methods,
  figure_title = "Absolute regret vs SAFE full: reference vs target decay families (SAFE variants)",
  output_file = "absolute_regret_reference_vs_target_decay_safe_methods.png"
)

make_drift_plot(
  summary_df[summary_df$reference_setting %in% powerlaw_reference & summary_df$target_setting %in% powerlaw_target, , drop = FALSE],
  row_order = powerlaw_reference,
  col_order = powerlaw_target,
  methods = names(method_labels),
  figure_title = "Reference drift: reference vs target powerlaw exponents",
  output_file = "reference_drift_reference_vs_target_powerlaw_all_methods.png"
)

make_drift_plot(
  summary_df[summary_df$reference_setting %in% powerlaw_reference & summary_df$target_setting %in% powerlaw_target, , drop = FALSE],
  row_order = powerlaw_reference,
  col_order = powerlaw_target,
  methods = safe_methods,
  figure_title = "Reference drift: reference vs target powerlaw exponents (SAFE variants)",
  output_file = "reference_drift_reference_vs_target_powerlaw_safe_methods.png"
)

make_regret_plot(
  summary_df[summary_df$reference_setting %in% powerlaw_reference & summary_df$target_setting %in% powerlaw_target, , drop = FALSE],
  row_order = powerlaw_reference,
  col_order = powerlaw_target,
  methods = names(method_labels),
  figure_title = "Absolute regret vs SAFE full: reference vs target powerlaw exponents",
  output_file = "absolute_regret_reference_vs_target_powerlaw_all_methods.png"
)

make_regret_plot(
  summary_df[summary_df$reference_setting %in% powerlaw_reference & summary_df$target_setting %in% powerlaw_target, , drop = FALSE],
  row_order = powerlaw_reference,
  col_order = powerlaw_target,
  methods = safe_methods,
  figure_title = "Absolute regret vs SAFE full: reference vs target powerlaw exponents (SAFE variants)",
  output_file = "absolute_regret_reference_vs_target_powerlaw_safe_methods.png"
)
