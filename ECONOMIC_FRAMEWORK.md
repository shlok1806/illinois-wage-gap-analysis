# Economic Framework: County-Level Wage Gap Analysis

## 1. Theoretical Framework

### 1.1 Mincer Earnings Equation with Discrimination
Building on the seminal work of Mincer (1974), we extend the human capital model to incorporate discrimination:

**Base Model:**
```
ln(w_i) = α + β₁S_i + β₂X_i + β₃Z_i + ε_i
```

**Extended Model with Discrimination:**
```
ln(w_i) = α + β₁S_i + β₂X_i + β₃Z_i + γ₁D_i + γ₂(D_i × S_i) + γ₃(D_i × X_i) + ε_i
```

Where:
- `w_i`: Individual earnings
- `S_i`: Years of schooling/education
- `X_i`: Experience and other human capital variables
- `Z_i`: County-level characteristics (demographics, industry mix)
- `D_i`: Discrimination indicator (gender, race)
- `γ₁`: Direct discrimination effect
- `γ₂`: Discrimination-education interaction
- `γ₃`: Discrimination-experience interaction

### 1.2 Taste-Based vs. Statistical Discrimination
Following Becker (1957) and Arrow (1973):

**Taste-Based Discrimination:**
- Employers have preferences against certain groups
- Discrimination coefficient: `d = (w_M - w_F)/w_M`
- Market forces may not eliminate discrimination due to imperfect competition

**Statistical Discrimination:**
- Employers use group averages as signals when individual productivity is uncertain
- `E[θ|G] = μ_G + σ_G²/(σ_G² + σ_ε²) × (s - μ_G)`
- Where θ is true productivity, G is group, s is signal

### 1.3 Spatial Equilibrium Model
Following Roback (1982) spatial equilibrium framework:

**Worker Utility:**
```
U = ln(w) - αln(r) + Q(Z)
```

**Firm Profit:**
```
π = p - w - r - C(Z)
```

**Spatial Equilibrium Conditions:**
- Workers indifferent across locations: `U_i = U_j`
- Firms earn zero profit: `π_i = π_j`
- Housing market clears: `H_i = N_i`

## 2. Literature Review

### 2.1 Key Papers

**1. Blau & Kahn (2017) - "The Gender Wage Gap: Extent, Trends, and Explanations"**
- **Findings**: Gender gap declined from 40% to 20% since 1980
- **Method**: Oaxaca-Blinder decomposition
- **Gap**: Focus on national trends, limited spatial analysis
- **Extension**: Our county-level analysis reveals spatial heterogeneity

**2. Autor et al. (2003) - "Computing Inequality: Have Computers Changed the Labor Market?"**
- **Findings**: Skill-biased technical change increased wage inequality
- **Method**: Instrumental variables with computer adoption
- **Gap**: Focus on skill premium, not gender/race gaps
- **Extension**: We examine how technology affects demographic wage gaps

**3. Card & Krueger (1992) - "School Quality and Black-White Relative Earnings"**
- **Findings**: School quality explains 20% of black-white wage gap
- **Method**: Natural experiment with school desegregation
- **Gap**: Focus on education quality, limited geographic scope
- **Extension**: We examine county-level education-demographic interactions

**4. Moretti (2004) - "Human Capital Externalities in Cities"**
- **Findings**: College graduates increase wages of less-educated workers
- **Method**: Instrumental variables with land-grant colleges
- **Gap**: Focus on human capital spillovers, not discrimination
- **Extension**: We examine how human capital affects demographic wage gaps

**5. Bertrand & Mullainathan (2004) - "Are Emily and Greg More Employable Than Lakisha and Jamal?"**
- **Findings**: Racial discrimination in callback rates
- **Method**: Audit study with randomized names
- **Gap**: Focus on hiring, not wage determination
- **Extension**: We examine wage discrimination in employment

### 2.2 Research Gaps Our Study Addresses
1. **Spatial Heterogeneity**: Most studies use national data, missing local variation
2. **Policy Relevance**: Limited analysis of county-level policy interventions
3. **Structural Interpretation**: Few studies estimate economic magnitudes
4. **Counterfactual Analysis**: Limited policy simulation work

## 3. Identification Strategy

### 3.1 Natural Experiments

**1. Illinois Minimum Wage Increases (2019-2025)**
- **Treatment**: Counties with different minimum wage levels
- **Control**: Counties with federal minimum wage
- **Identification**: `E[Δw|T=1] - E[Δw|T=0]`
- **Assumption**: Parallel trends in absence of treatment

**2. Manufacturing Decline (2000-2020)**
- **Treatment**: Counties with large manufacturing employment
- **Control**: Counties with diverse employment
- **Identification**: Differential impact on demographic groups
- **Instrument**: National manufacturing trends × County manufacturing share

**3. Education Policy Changes**
- **Treatment**: Counties with different school funding formulas
- **Control**: Counties with stable funding
- **Identification**: Changes in education-demographic wage relationships

### 3.2 Instrumental Variables

**1. Distance to Land-Grant Colleges**
- **Instrument**: Distance to nearest land-grant college
- **Exclusion Restriction**: Distance affects education but not wages directly
- **First Stage**: `Education = α + β₁Distance + β₂Controls + ε`
- **Second Stage**: `ln(Wage) = γ + δ₁Education_hat + δ₂Controls + ν`

**2. Historical Railroad Networks**
- **Instrument**: Distance to historical railroad lines
- **Exclusion Restriction**: Historical infrastructure affects current demographics but not wages
- **Use**: Instrument for demographic composition

**3. Climate Variation**
- **Instrument**: Temperature/precipitation variation
- **Exclusion Restriction**: Climate affects industry mix but not individual productivity
- **Use**: Instrument for industry composition

### 3.3 Fixed Effects Specifications

**County-Year Fixed Effects:**
```
ln(w_ict) = α + β₁X_ict + β₂Z_ct + γ_c + δ_t + ε_ict
```

**County-Demographic Fixed Effects:**
```
ln(w_ict) = α + β₁X_ict + β₂Z_ct + γ_cg + δ_t + ε_ict
```

Where:
- `c`: County
- `t`: Time
- `g`: Demographic group
- `γ_c`: County fixed effects
- `γ_cg`: County-demographic fixed effects

## 4. Structural Interpretation

### 4.1 Elasticity Calculations

**Education Elasticity:**
```
η_education = ∂ln(w)/∂S = β₁
```

**Experience Elasticity:**
```
η_experience = ∂ln(w)/∂X = β₂
```

**Discrimination Elasticity:**
```
η_discrimination = ∂ln(w)/∂D = γ₁
```

### 4.2 Welfare Implications

**Deadweight Loss from Discrimination:**
```
DWL = 0.5 × (w_competitive - w_discriminatory) × (L_competitive - L_discriminatory)
```

**Consumer Surplus Loss:**
```
CS_loss = ∫[D(p) - D(p_discriminatory)]dp
```

**Producer Surplus Loss:**
```
PS_loss = ∫[S(p) - S(p_discriminatory)]dp
```

### 4.3 Counterfactual Scenarios

**1. Perfect Competition:**
- Eliminate discrimination coefficient: `d = 0`
- Recalculate wage distribution
- Estimate welfare gains

**2. Equal Education Access:**
- Set education levels equal across groups
- Estimate wage convergence
- Calculate required investment

**3. Policy Interventions:**
- Pay transparency laws
- Anti-discrimination enforcement
- Education subsidies

## 5. Policy Analysis

### 5.1 Policy Simulation Framework

**1. Pay Transparency Laws**
- **Model**: `w_post = w_pre + β × Transparency_Index`
- **Estimation**: Use variation in state-level transparency laws
- **Prediction**: County-level wage convergence

**2. Anti-Discrimination Enforcement**
- **Model**: `w_post = w_pre + γ × Enforcement_Intensity`
- **Estimation**: Use EEOC enforcement data
- **Prediction**: Reduction in discrimination coefficient

**3. Education Subsidies**
- **Model**: `S_post = S_pre + δ × Subsidy_Amount`
- **Estimation**: Use variation in Pell Grant eligibility
- **Prediction**: Long-term wage convergence

### 5.2 Cost-Benefit Analysis

**Benefits:**
- Increased labor force participation
- Higher productivity
- Reduced inequality
- Social welfare improvements

**Costs:**
- Policy implementation costs
- Administrative burden
- Potential efficiency losses
- Transition costs

**Net Present Value:**
```
NPV = Σ(B_t - C_t)/(1 + r)^t
```

### 5.3 Distributional Effects

**Lorenz Curve Analysis:**
- Pre-policy income distribution
- Post-policy income distribution
- Gini coefficient changes

**Quantile Treatment Effects:**
```
QTE(τ) = F_w1^(-1)(τ) - F_w0^(-1)(τ)
```

**Poverty Impact:**
- Headcount ratio changes
- Poverty gap changes
- Severity index changes

## 6. Empirical Implementation

### 6.1 Data Requirements

**Individual-Level Data:**
- CPS, ACS, or administrative records
- Earnings, demographics, education
- Geographic identifiers

**County-Level Data:**
- Economic indicators
- Policy variables
- Demographic composition

**Instrumental Variables:**
- Historical data
- Geographic features
- Policy variation

### 6.2 Estimation Strategy

**1. Reduced Form:**
```
ln(w_ict) = α + β₁Policy_ct + β₂X_ict + β₃Z_ct + γ_c + δ_t + ε_ict
```

**2. Structural Model:**
```
ln(w_ict) = α + β₁S_ict + β₂X_ict + β₃Z_ct + γ₁D_ict + γ₂(D_ict × S_ict) + ε_ict
```

**3. Policy Simulation:**
```
w_counterfactual = exp(α_hat + β_hat₁S_new + β_hat₂X_ict + β_hat₃Z_ct + γ_hat₁D_new)
```

### 6.3 Robustness Checks

**1. Specification Tests:**
- Ramsey RESET test
- Hausman test for endogeneity
- Overidentification tests

**2. Sensitivity Analysis:**
- Different sample periods
- Alternative specifications
- Various control sets

**3. Placebo Tests:**
- Falsification exercises
- Pre-treatment trends
- Alternative outcomes

This framework provides a rigorous economic foundation for the county-level wage gap analysis, moving beyond descriptive statistics to causal inference and policy evaluation. 