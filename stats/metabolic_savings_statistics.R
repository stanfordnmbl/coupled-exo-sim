require("lme4")
require("multcomp")
require("minqa")

citation()
citation("lme4")
citation("multcomp")

verbosity = 1

# Redirect all output to a text file.
sink("metabolic_savings_statistics.txt", append=FALSE, split=TRUE)

perform_stats_for_formula <- function(exp, formula) {
    print("");
    print("");
    print("=========================== formula: =============================")
    print(formula)
    print("==================================================================")
    model = lme4::lmer(formula, data=exp.data)
    if (verbosity == 1) {
        print(summary(model))
        print(confint(model))
    }
    z <- anova(model)
    str(z)
    # Perform pairwise comparisons.
    comparisons = multcomp::glht(model, linfct=mcp(mod="Tukey"))
    if (verbosity == 1) {
        print(summary(comparisons,
                      ))
    }
    # TODO adjust pvalues?
    # summary(comparisons, test=adjusted(type="bonferroni"))
    compact_letter_display = multcomp::cld(comparisons, decreasing=TRUE)
    # Letters show which levels are not significantly different from each
    # other. Levels are labeled in decreasing order (e.g., the greatest level
    # has the letter 'a').
    print(compact_letter_display)
}

filepaths <- c('metabolics_for_stats_percent_savings.csv')
for (filepath in filepaths) {

    print("");
    print("");
    print("=========================== filename: ============================")
    print(filepath)
    print("==================================================================")

    exp.data = read.csv(filepath, fileEncoding="UTF-8-BOM")
    str(exp.data)
    
    # Set certain columns as categorical.
    # Convert "percent increase" into "percent savings."
    exp.data$savings <- -exp.data$savings
    exp.data$mod <- as.factor(exp.data$mod)
    exp.data$subject <- as.factor(exp.data$subject)
    # exp.data$base_sim <- as.factor(paste(as.character(exp.data$subject),
    #                                  rep(c("1", "2"), 12),
    #                                  sep=","))

    str(exp.data)

    # Average over trials.
    exp_by_subj.data <- aggregate(. ~ subject + mod, data=exp.data, mean)
    perform_stats_for_formula(exp_by_subj, savings ~ mod + (1 | subject))
}

# Did all of the devices decrease metabolic cost?
filepath <- 'metabolics_for_stats_absolute_values.csv'
print("");
print("");
print("=========================== filename: ============================")
print(filepath)
print("==================================================================")
exp.data = read.csv(filepath, fileEncoding="UTF-8-BOM")
exp.data$mod <- as.factor(exp.data$mod)
exp.data$subject <- as.factor(exp.data$subject)
str(exp.data)
# Make "none" the reference level for the mod factor.
exp.data <- within(exp.data, mod <- relevel(mod, ref="none"))
exp_by_subj.data <- aggregate(. ~ subject + mod, data=exp.data, mean)
perform_stats_for_formula(exp_by_subj, metcost ~ mod + (1 | subject))


sink()