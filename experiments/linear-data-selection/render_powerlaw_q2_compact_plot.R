#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(patchwork)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: Rscript render_powerlaw_q2_compact_plot.R <output_dir>")
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

all_methods <- names(method_labels)

make_offsets <- function(methods) {
  if (length(methods) == 1) {
    setNames(0.0, methods)
  } else {
    setNames(seq(-0.14, 0.14, length.out = length(methods)), methods)
  }
}

prepare_df <- function(df, methods) {
  plot_df <- df[df$method %in% methods, , drop = FALSE]
  methods <- methods[methods %in% unique(plot_df$method)]
  methods <- methods[order(match(methods, names(method_labels)))]
  offsets <- make_offsets(methods)
  x_levels <- sort(unique(plot_df$K_requested))
  x_map <- setNames(seq_along(x_levels), as.character(x_levels))

  plot_df$method <- factor(plot_df$method, levels = methods)
  plot_df$method_label <- factor(method_labels[as.character(plot_df$method)], levels = method_labels[methods])
  plot_df$x_base <- unname(x_map[as.character(plot_df$K_requested)])
  plot_df$x_plot <- plot_df$x_base + unname(offsets[as.character(plot_df$method)])
  plot_df$regret_ymin <- plot_df$absolute_regret_mean - plot_df$absolute_regret_sem
  plot_df$regret_ymax <- plot_df$absolute_regret_mean + plot_df$absolute_regret_sem
  plot_df$drift_ymin <- pmax(0.1, plot_df$drift_mean - plot_df$drift_sem)
  plot_df$drift_ymax <- pmax(plot_df$drift_mean + plot_df$drift_sem, plot_df$drift_ymin * 1.001)

  list(
    df = plot_df,
    methods = methods,
    x_levels = x_levels,
    x_breaks = seq_along(x_levels),
    x_labels = as.character(x_levels),
    color_values = setNames(unname(method_colors[methods]), method_labels[methods])
  )
}

base_theme <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "grey85", linewidth = 0.3),
    panel.grid.major.y = element_line(color = "grey80", linewidth = 0.35),
    legend.position = "bottom",
    legend.box = "horizontal",
    axis.text = element_text(size = 12, color = "black"),
    axis.title = element_text(size = 13, color = "black"),
    legend.text = element_text(size = 11),
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )

make_compact_plot <- function(df, methods, title_prefix, output_file) {
  prepared <- prepare_df(df, methods)
  plot_df <- prepared$df
  y_upper_drift <- max(plot_df$drift_ymax, na.rm = TRUE)
  y_upper_drift <- max(1.2, y_upper_drift * 1.08)

  p_regret <- ggplot(plot_df, aes(x = x_plot, y = absolute_regret_mean, color = method_label, group = method_label)) +
    geom_line(linewidth = 0.65, alpha = 0.92) +
    geom_errorbar(aes(ymin = regret_ymin, ymax = regret_ymax), width = 0.04, linewidth = 0.45, alpha = 0.85) +
    geom_point(size = 1.7, alpha = 0.98) +
    scale_color_manual(values = prepared$color_values, breaks = method_labels[prepared$methods], drop = FALSE) +
    scale_x_continuous(breaks = prepared$x_breaks, labels = prepared$x_labels) +
    scale_y_continuous(trans = pseudo_log_trans(base = 10, sigma = 0.05)) +
    geom_hline(yintercept = 0.0, color = "#666666", linetype = "dashed", linewidth = 0.5) +
    labs(
      title = "Absolute Regret",
      x = "Requested rank K_R + K_T",
      y = "Absolute regret vs SAFE full",
      color = NULL
    ) +
    base_theme

  p_drift <- ggplot(plot_df, aes(x = x_plot, y = drift_mean, color = method_label, group = method_label)) +
    geom_line(linewidth = 0.65, alpha = 0.92) +
    geom_errorbar(aes(ymin = drift_ymin, ymax = drift_ymax), width = 0.04, linewidth = 0.45, alpha = 0.85) +
    geom_point(size = 1.7, alpha = 0.98) +
    scale_color_manual(values = prepared$color_values, breaks = method_labels[prepared$methods], drop = FALSE) +
    scale_x_continuous(breaks = prepared$x_breaks, labels = prepared$x_labels) +
    scale_y_log10(limits = c(0.1, y_upper_drift)) +
    labs(
      title = "Reference Drift",
      x = "Requested rank K_R + K_T",
      y = "Mean reference drift",
      color = NULL
    ) +
    base_theme

  if (!is.null(rho) && !is.na(rho)) {
    p_drift <- p_drift + geom_hline(yintercept = rho, color = "#c62828", linetype = "dashed", linewidth = 0.5)
  }

  combined <- (p_regret + p_drift + plot_layout(ncol = 2, guides = "collect")) &
    theme(legend.position = "bottom")

  ggsave(
    filename = file.path(output_dir, output_file),
    plot = combined + plot_annotation(title = title_prefix),
    width = 13.8,
    height = 6.6,
    dpi = 180,
    bg = "white"
  )
}

q2_df <- summary_df[
  summary_df$reference_setting == "powerlaw-q2" &
    summary_df$target_setting == "powerlaw-q2",
  ,
  drop = FALSE
]

make_compact_plot(
  q2_df,
  safe_methods,
  "Power-law spectra with q_R = q_T = 2 (SAFE variants)",
  "compact_q2_regret_drift_safe_methods.png"
)

make_compact_plot(
  q2_df,
  all_methods,
  "Power-law spectra with q_R = q_T = 2 (all methods)",
  "compact_q2_regret_drift_all_methods.png"
)
