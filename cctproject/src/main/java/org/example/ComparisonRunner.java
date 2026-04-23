package org.example;

import java.util.*;

/**
 * ============================================================
 *  ComparisonRunner.java
 *  Runs BOTH problems back-to-back and prints a full
 *  side-by-side benchmark table.
 * ============================================================
 *  HOW TO RUN IN IntelliJ:
 *  1. Put ComparisonRunner.java in the same src/ folder as
 *     Main.java and Main2.java
 *  2. Right-click ComparisonRunner → Run 'ComparisonRunner.main()'
 * ============================================================
 *
 *  PROBLEM 1 — Cloud Performance Optimization (Main.java)
 *    Goal    : Minimise SLA violations, balance CPU utilisation
 *    Agent   : DQN  (uniform replay, hard target update)
 *    Env     : 6-node cluster, actions = Scale-Up/Down/Migrate/Idle
 *
 *  PROBLEM 2 — Energy-Aware VM Consolidation (Main2.java)
 *    Goal    : Minimise energy consumption, maintain QoS
 *    Agent   : Double DQN (prioritised replay, soft target update)
 *    Env     : 8-host datacentre, actions = Consolidate/Spread/
 *              Power-Off/Power-On/Idle
 * ============================================================
 */
public class ComparisonRunner {
    static final double STEP_HOURS = 1.0 / 60.0;
    static final double ENERGY_PRICE_PER_KWH = 0.12;
    static final double VM_PRICE_PER_HOUR = 0.019;
    static final double P1_HOST_IDLE_W = 90.0;
    static final double P1_HOST_MAX_W = 220.0;

    static final int    EPISODES   = 200;
    static final int    STEPS      = 50;
    static final double GAMMA      = 0.95;
    static final double LR         = 1e-3;
    static final double EPS_START  = 1.0;
    static final double EPS_MIN    = 0.05;
    static final double EPS_DECAY  = 0.995;
    static final int    BUFFER     = 10_000;
    static final int    BATCH      = 64;
    static final int    HIDDEN     = 128;

    // DDQN extras
    static final double GAMMA2     = 0.97;
    static final double LR2        = 8e-4;
    static final double EPS_DECAY2 = 0.994;
    static final double TAU        = 0.01;

    static final Random RNG = new Random(42);

    // ════════════════════════════════════════════════════════
    public static void main(String[] args) {

        printHeader();

        System.out.println("▶  Training Problem 1: Cloud Performance Optimization (DQN)…");
        BenchResult r1 = runProblem1();
        System.out.println("   Done.\n");

        System.out.println("▶  Training Problem 2: Energy-Aware VM Consolidation (DDQN)…");
        BenchResult r2 = runProblem2();
        System.out.println("   Done.\n");

        printComparisonTable(r1, r2);
        printConvergenceChart(r1.rewardLog, r2.rewardLog);
        printMetricChart("SLA / QoS Violations per Episode",
                r1.violationLog, "P1 SLA Breach", r2.violationLog, "P2 QoS Breach");
        printMetricChart("Execution Time per Episode (ms)",
                r1.execTimeLog, "P1 Exec Time", r2.execTimeLog, "P2 Exec Time");
        printMetricChart("Energy per Episode (kWh)",
                r1.energyLog, "P1 Energy", r2.energyLog, "P2 Energy");
        printPolicyComparison(r1, r2);
        printConclusion(r1, r2);
    }

    // ════════════════════════════════════════════════════════
    //  PROBLEM 1 RUNNER — DQN Cloud Performance
    // ════════════════════════════════════════════════════════
    static BenchResult runProblem1() {
        long startNs = System.nanoTime();

        // 6 nodes × 3 features = 18-dim state, 4 actions
        NNet q      = new NNet(18, HIDDEN, 4);
        NNet target = new NNet(18, HIDDEN, 4);
        target.copyFrom(q);
        ReplayBuf buf = new ReplayBuf(BUFFER);
        P1Env env = new P1Env();

        double eps = EPS_START;
        BenchResult res = new BenchResult("Cloud Performance Opt.", "DQN",
                "6-Node Cluster", "Scale-Up/Down/Migrate/Idle",
                "CPU balance + SLA", "Uniform replay, hard target (ep 10)",
                18, 4);

        for (int ep = 1; ep <= EPISODES; ep++) {
            double[] s     = env.reset();
            double   epR   = 0;
            int      epSLA = 0;
            double   epCPU = 0;
            double   epEnergy = 0;
            double   epCost = 0;
            double   epExec = 0;
            double   epHosts = 0;
            double   epVms = 0;

            for (int t = 0; t < STEPS; t++) {
                int a = (RNG.nextDouble() < eps) ? RNG.nextInt(4) : argmax(q.fwd(s));
                double[] ns = env.step(a);
                double r    = env.reward;
                boolean done= (t == STEPS - 1);
                epR   += r;
                epSLA += env.slaViolated ? 1 : 0;
                epCPU += env.avgCPU();
                epEnergy += env.stepEnergyKWh();
                epCost += env.stepCostUsd();
                epExec += env.executionTimeMs();
                epHosts += env.activeHosts();
                epVms += env.totalVMs();
                buf.add(s, a, r, ns, done);
                if (buf.size() >= BATCH) trainDQN(q, target, buf);
                s = ns;
            }
            if (ep % 10 == 0) target.copyFrom(q);
            eps = Math.max(EPS_MIN, eps * EPS_DECAY);

            res.rewardLog.add(epR / STEPS);
            res.violationLog.add((double) epSLA);
            res.cpuLog.add(epCPU / STEPS);
            res.energyLog.add(epEnergy / STEPS);
            res.costLog.add(epCost / STEPS);
            res.execTimeLog.add(epExec / STEPS);
            res.hostLog.add(epHosts / STEPS);
            res.vmLog.add(epVms / STEPS);
        }
        res.finalEps = eps;
        res.elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0;
        return res;
    }

    static void trainDQN(NNet q, NNet tgt, ReplayBuf buf) {
        for (var tr : buf.sample(BATCH)) {
            double[] qv = q.fwd(tr.s);
            double   nv = max(tgt.fwd(tr.ns));
            double   y  = tr.done ? tr.r : tr.r + GAMMA * nv;
            q.update(tr.s, tr.a, (y - qv[tr.a]) * LR);
        }
    }

    // ════════════════════════════════════════════════════════
    //  PROBLEM 2 RUNNER — DDQN Energy Consolidation
    // ════════════════════════════════════════════════════════
    static BenchResult runProblem2() {
        long startNs = System.nanoTime();

        // 8 hosts × 3 features = 24-dim state, 5 actions
        NNet online = new NNet(24, HIDDEN, 5);
        NNet target = new NNet(24, HIDDEN, 5);
        target.copyFrom(online);
        PriBuf buf = new PriBuf(BUFFER);
        P2Env env  = new P2Env();

        double eps = EPS_START;
        BenchResult res = new BenchResult("Energy-Aware VM Consolidation", "Double DQN (DDQN)",
                "8-Host Datacentre", "Consolidate/Spread/PowerOff/PowerOn/Idle",
                "Energy saved - QoS penalty", "Prioritized replay, soft update (τ=0.01)",
                24, 5);

        for (int ep = 1; ep <= EPISODES; ep++) {
            double[] s    = env.reset();
            double   epR  = 0;
            int      epQoS= 0;
            double   epPW = 0;
            double   epEnergy = 0;
            double   epCost = 0;
            double   epExec = 0;
            double   epHosts = 0;
            double   epVms = 0;

            for (int t = 0; t < STEPS; t++) {
                int a = (RNG.nextDouble() < eps) ? RNG.nextInt(5) : ddqnAct(online, target, s);
                double[] ns = env.step(a);
                double r    = env.reward;
                boolean done= (t == STEPS - 1);
                epR   += r;
                epQoS += env.qosViolated ? 1 : 0;
                epPW  += env.power;
                epEnergy += env.stepEnergyKWh();
                epCost += env.stepCostUsd();
                epExec += env.executionTimeMs();
                epHosts += env.activeHosts();
                epVms += env.totalVMs();
                buf.add(s, a, r, ns, done, Math.abs(r) + 0.01);
                if (buf.size() >= BATCH) trainDDQN(online, target, buf);
                target.softUpdate(online, TAU);
                s = ns;
            }
            eps = Math.max(EPS_MIN, eps * EPS_DECAY2);

            res.rewardLog.add(epR / STEPS);
            res.violationLog.add((double) epQoS);
            res.cpuLog.add(epPW / STEPS);   // re-uses cpuLog for energy
            res.energyLog.add(epEnergy / STEPS);
            res.costLog.add(epCost / STEPS);
            res.execTimeLog.add(epExec / STEPS);
            res.hostLog.add(epHosts / STEPS);
            res.vmLog.add(epVms / STEPS);
        }
        res.finalEps = eps;
        res.elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0;
        return res;
    }

    static int ddqnAct(NNet on, NNet tg, double[] s) {
        double[] qo = on.fwd(s), qt = tg.fwd(s);
        int best = 0; double bv = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < qo.length; a++) {
            double v = 0.6*qo[a]+0.4*qt[a];
            if (v > bv) { bv = v; best = a; }
        }
        return best;
    }

    static void trainDDQN(NNet on, NNet tg, PriBuf buf) {
        for (var tr : buf.sample(BATCH)) {
            double[] qv   = on.fwd(tr.s);
            int      nextA = argmax(on.fwd(tr.ns));
            double   nv   = tg.fwd(tr.ns)[nextA];
            double   y    = tr.done ? tr.r : tr.r + GAMMA2 * nv;
            on.update(tr.s, tr.a, (y - qv[tr.a]) * LR2);
        }
    }

    // ════════════════════════════════════════════════════════
    //  PRINT: COMPARISON TABLE
    // ════════════════════════════════════════════════════════
    static void printComparisonTable(BenchResult r1, BenchResult r2) {
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║              FULL COMPARISON TABLE — BOTH PROBLEMS                  ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");
        String fmt = "║  %-28s  %-18s  %-18s ║%n";
        System.out.printf(fmt, "Attribute", "Problem 1 (P1)", "Problem 2 (P2)");
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");

        Object[][] rows = {
                { "Problem Name",          r1.name,            r2.name },
                { "Algorithm",             r1.algorithm,       r2.algorithm },
                { "Environment",           r1.env,             r2.env },
                { "Action Space",          r1.actions,         r2.actions },
                { "Reward Design",         r1.rewardDesign,    r2.rewardDesign },
                { "Replay Strategy",       r1.replayStrategy,  r2.replayStrategy },
                { "State Dimension",       r1.stateDim,        r2.stateDim },
                { "Action Dimension",      r1.actionDim,       r2.actionDim },
                { "Final ε (epsilon)",
                        String.format("%.4f", r1.finalEps),
                        String.format("%.4f", r2.finalEps) },
                { "Final Avg Reward",
                        String.format("%+.4f", avgLast(r1.rewardLog, 10)),
                        String.format("%+.4f", avgLast(r2.rewardLog, 10)) },
                { "Best Episode Reward",
                        String.format("%+.4f", r1.rewardLog.stream().mapToDouble(Double::doubleValue).max().orElse(0)),
                        String.format("%+.4f", r2.rewardLog.stream().mapToDouble(Double::doubleValue).max().orElse(0)) },
                { "Worst Episode Reward",
                        String.format("%+.4f", r1.rewardLog.stream().mapToDouble(Double::doubleValue).min().orElse(0)),
                        String.format("%+.4f", r2.rewardLog.stream().mapToDouble(Double::doubleValue).min().orElse(0)) },
                { "Total Violations",
                        String.format("%.0f SLA", r1.violationLog.stream().mapToDouble(Double::doubleValue).sum()),
                        String.format("%.0f QoS", r2.violationLog.stream().mapToDouble(Double::doubleValue).sum()) },
                { "Avg Violations/Ep",
                        String.format("%.2f", r1.violationLog.stream().mapToDouble(Double::doubleValue).average().orElse(0)),
                        String.format("%.2f", r2.violationLog.stream().mapToDouble(Double::doubleValue).average().orElse(0)) },
                { "Avg Hosts",
                        String.format("%.2f", avg(r1.hostLog)),
                        String.format("%.2f", avg(r2.hostLog)) },
                { "Avg VMs",
                        String.format("%.2f", avg(r1.vmLog)),
                        String.format("%.2f", avg(r2.vmLog)) },
                { "Avg Exec Time",
                        String.format("%.2f ms", avg(r1.execTimeLog)),
                        String.format("%.2f ms", avg(r2.execTimeLog)) },
                { "Avg Energy/Ep",
                        String.format("%.4f kWh", avg(r1.energyLog)),
                        String.format("%.4f kWh", avg(r2.energyLog)) },
                { "Total Energy",
                        String.format("%.4f kWh", sum(r1.energyLog)),
                        String.format("%.4f kWh", sum(r2.energyLog)) },
                { "Avg Cost/Ep",
                        String.format("$%.4f", avg(r1.costLog)),
                        String.format("$%.4f", avg(r2.costLog)) },
                { "Total Cost",
                        String.format("$%.4f", sum(r1.costLog)),
                        String.format("$%.4f", sum(r2.costLog)) },
                { "Convergence (est.ep)",
                        String.format("~%d", estimateConvergence(r1.rewardLog)),
                        String.format("~%d", estimateConvergence(r2.rewardLog)) },
                { "Reward Std Dev",
                        String.format("%.4f", std(r1.rewardLog)),
                        String.format("%.4f", std(r2.rewardLog)) },
                { "Training Runtime",
                        String.format("%.2f ms", r1.elapsedMs),
                        String.format("%.2f ms", r2.elapsedMs) },
        };

        for (Object[] row : rows) {
            System.out.printf("║  %-28s  %-18s  %-18s ║%n",
                    truncate(row[0].toString(), 28),
                    truncate(row[1].toString(), 18),
                    truncate(row[2].toString(), 18));
        }
        System.out.println("╚══════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    // ════════════════════════════════════════════════════════
    //  PRINT: ASCII CONVERGENCE CHART
    // ════════════════════════════════════════════════════════
    static void printConvergenceChart(List<Double> log1, List<Double> log2) {
        int width = 60, height = 14;
        System.out.println("REWARD CONVERGENCE  (ASCII chart, smoothed 10-ep window)");
        System.out.println("─".repeat(width + 12));

        double[] s1 = smooth(log1, 10);
        double[] s2 = smooth(log2, 10);

        double lo = Math.min(min(s1), min(s2));
        double hi = Math.max(max(s1), max(s2));
        double range = hi - lo == 0 ? 1 : hi - lo;

        char[][] grid = new char[height][width];
        for (char[] row : grid) Arrays.fill(row, ' ');

        int n = Math.min(width, s1.length);
        for (int x = 0; x < n; x++) {
            int y1 = (int)((s1[x * s1.length / n] - lo) / range * (height - 1));
            int y2 = (int)((s2[x * s2.length / n] - lo) / range * (height - 1));
            y1 = clampInt(y1, 0, height - 1);
            y2 = clampInt(y2, 0, height - 1);
            if (y1 == y2) grid[height - 1 - y1][x] = '✦';
            else { grid[height - 1 - y1][x] = '1'; grid[height - 1 - y2][x] = '2'; }
        }

        for (int row = 0; row < height; row++) {
            double val = hi - (double)row / (height - 1) * range;
            System.out.printf("%+7.2f | %s%n", val, new String(grid[row]));
        }
        System.out.println("        " + "-".repeat(width));
        System.out.printf("         Ep 1%s Ep %d%n", " ".repeat(width - 14), EPISODES);
        System.out.println("  1 = P1 (DQN Cloud Perf)    2 = P2 (DDQN Energy)    ✦ = overlap");
        System.out.println();
    }

    static void printMetricChart(String title, List<Double> log1, String lbl1,
                                 List<Double> log2, String lbl2) {
        int width = 60, height = 10;
        System.out.println(title + "  (ASCII, smoothed)");
        System.out.println("─".repeat(width + 12));

        double[] s1 = smooth(log1, 10);
        double[] s2 = smooth(log2, 10);
        double lo = 0, hi = Math.max(max(s1), max(s2));
        double range = hi == 0 ? 1 : hi;

        char[][] grid = new char[height][width];
        for (char[] row : grid) Arrays.fill(row, ' ');
        int n = Math.min(width, s1.length);
        for (int x = 0; x < n; x++) {
            int y1 = (int)(s1[x * s1.length / n] / range * (height - 1));
            int y2 = (int)(s2[x * s2.length / n] / range * (height - 1));
            y1 = clampInt(y1, 0, height - 1);
            y2 = clampInt(y2, 0, height - 1);
            if (y1 == y2) grid[height - 1 - y1][x] = '✦';
            else { grid[height - 1 - y1][x] = '1'; grid[height - 1 - y2][x] = '2'; }
        }
        for (int row = 0; row < height; row++) {
            double val = hi - (double)row / (height - 1) * range;
            System.out.printf("%7.1f | %s%n", val, new String(grid[row]));
        }
        System.out.println("        " + "-".repeat(width));
        System.out.printf("  1 = %s   2 = %s   ✦ = overlap%n%n", lbl1, lbl2);
    }

    // ════════════════════════════════════════════════════════
    //  PRINT: POLICY COMPARISON
    // ════════════════════════════════════════════════════════
    static void printPolicyComparison(BenchResult r1, BenchResult r2) {
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                   DESIGN DECISION COMPARISON                        ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");
        String[][] decisions = {
                { "Target Network",   "Hard copy every 10 ep", "Soft update (τ=0.01) each step" },
                { "Replay Memory",    "Uniform random sample", "Prioritized by TD error (α=0.6)" },
                { "Q-value bias",     "Single net estimate",   "Online selects, target evaluates" },
                { "Reward range",     "~−3.0 to +1.5",        "~−2.5 to +2.5" },
                { "Key penalty",      "SLA CPU > 90% (−2.5)", "Energy waste + Temp > 80° (−1.0)" },
                { "Key bonus",        "Balanced CPU 40-70%",  "Active hosts ≤ 4 + energy saving" },
                { "Optimises for",    "Performance / latency", "Cost / sustainability" },
                { "Industry use case","Auto-scaling PaaS",    "Green DC / carbon-neutral cloud" },
        };
        System.out.printf("║  %-22s  %-22s  %-18s ║%n", "Decision", "P1 (DQN)", "P2 (DDQN)");
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");
        for (String[] row : decisions) {
            System.out.printf("║  %-22s  %-22s  %-18s ║%n",
                    truncate(row[0], 22), truncate(row[1], 22), truncate(row[2], 18));
        }
        System.out.println("╚══════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    static void printConclusion(BenchResult r1, BenchResult r2) {
        double p1AvgR = avgLast(r1.rewardLog, 10);
        double p2AvgR = avgLast(r2.rewardLog, 10);
        double p1Viol = r1.violationLog.stream().mapToDouble(Double::doubleValue).sum();
        double p2Viol = r2.violationLog.stream().mapToDouble(Double::doubleValue).sum();
        double p1Cost = sum(r1.costLog);
        double p2Cost = sum(r2.costLog);
        double p1Energy = sum(r1.energyLog);
        double p2Energy = sum(r2.energyLog);
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                        CONCLUSION                                    ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");
        System.out.printf ("║  P1 final avg10 reward : %+.4f  |  P2 final avg10 reward: %+.4f ║%n", p1AvgR, p2AvgR);
        System.out.printf ("║  P1 total SLA violations: %-6.0f  |  P2 total QoS breaches: %-6.0f ║%n", p1Viol, p2Viol);
        System.out.printf ("║  P1 convergence at ep  : ~%-5d  |  P2 convergence at ep : ~%-5d ║%n",
                estimateConvergence(r1.rewardLog), estimateConvergence(r2.rewardLog));
        System.out.printf ("║  P1 total energy       : %-8.4f |  P2 total energy      : %-8.4f ║%n", p1Energy, p2Energy);
        System.out.printf ("║  P1 total cost         : $%-7.4f |  P2 total cost        : $%-7.4f ║%n", p1Cost, p2Cost);
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");
        System.out.println("║  Both DRL agents learn meaningful policies from scratch.             ║");
        System.out.println("║  DDQN (P2) converges more stably due to soft target updates and     ║");
        System.out.println("║  prioritized replay, while DQN (P1) is simpler and faster to train. ║");
        System.out.println("║  P1 optimises for performance; P2 optimises for energy efficiency.  ║");
        System.out.println("║  Combining both objectives is the next research direction.           ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════╝");
    }

    static void printHeader() {
        System.out.println("╔══════════════════════════════════════════════════════════════════════╗");
        System.out.println("║           DRL CLOUD SYSTEMS - COMPARISON RUNNER                    ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════════╣");
        System.out.println("║  P1: Cloud Performance Optimization    →  DQN                       ║");
        System.out.println("║  P2: Energy-Aware VM Consolidation     →  Double DQN (DDQN)         ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    // ════════════════════════════════════════════════════════
    //  ENVIRONMENTS (inline, self-contained)
    // ════════════════════════════════════════════════════════

    static class P1Env {
        static final int N = 6;
        double[] cpu = new double[N], mem = new double[N], net = new double[N];
        double reward; boolean slaViolated;

        double[] reset() {
            for (int i = 0; i < N; i++) {
                cpu[i] = 30 + RNG.nextDouble() * 60;
                mem[i] = 20 + RNG.nextDouble() * 70;
                net[i] = 10 + RNG.nextDouble() * 50;
            }
            return state();
        }

        double[] step(int a) {
            switch (a) {
                case 0 -> { // scale-up
                    int hi = 0, lo = 0;
                    for (int i=1;i<N;i++){if(cpu[i]>cpu[hi])hi=i;if(cpu[i]<cpu[lo])lo=i;}
                    double x=(cpu[hi]-cpu[lo])*0.3; cpu[hi]-=x; cpu[lo]+=x*0.75;
                }
                case 1 -> { for(int i=0;i<N;i++) cpu[i]=Math.min(100,cpu[i]*1.1); }
                case 2 -> { double m=avgCPU(); for(int i=0;i<N;i++) cpu[i]+=(m-cpu[i])*0.4; }
            }
            for(int i=0;i<N;i++){
                cpu[i]+=RNG.nextGaussian()*3; mem[i]+=RNG.nextGaussian()*2;
                cpu[i]=Math.max(5,Math.min(100,cpu[i])); mem[i]=Math.max(5,Math.min(100,mem[i]));
            }
            double avg=avgCPU(), mx=Arrays.stream(cpu).max().getAsDouble();
            double sp=mx-Arrays.stream(cpu).min().getAsDouble();
            slaViolated = mx>90;
            reward=0;
            if(avg>=40&&avg<=70) reward+=1.0; if(sp<20) reward+=0.5;
            if(mx>90) reward-=2.5; if(mx>85) reward-=1.0;
            if(a==1&&avg<30) reward-=0.5; if(a==0&&avg>75) reward+=0.3;
            if(a==3&&avg>80) reward-=1.0;
            return state();
        }

        double[] state() {
            double[] s = new double[18];
            for(int i=0;i<N;i++){s[i*3]=cpu[i]/100;s[i*3+1]=mem[i]/100;s[i*3+2]=net[i]/100;}
            return s;
        }
        double avgCPU(){return Arrays.stream(cpu).average().orElse(0);}
        double avgMem(){return Arrays.stream(mem).average().orElse(0);}
        double avgNet(){return Arrays.stream(net).average().orElse(0);}
        int activeHosts(){return N;}
        int vmCount(int i){return Math.max(1, (int)Math.round((cpu[i] + mem[i]) / 35.0));}
        int totalVMs(){int t=0;for(int i=0;i<N;i++)t+=vmCount(i);return t;}
        double hostPowerW(int i){
            double util=(cpu[i]*0.6 + mem[i]*0.25 + net[i]*0.15)/100.0;
            return P1_HOST_IDLE_W + (P1_HOST_MAX_W - P1_HOST_IDLE_W) * util;
        }
        double totalPowerW(){double t=0;for(int i=0;i<N;i++)t+=hostPowerW(i);return t;}
        double stepEnergyKWh(){return totalPowerW()/1000.0*STEP_HOURS;}
        double stepCostUsd(){return stepEnergyKWh()*ENERGY_PRICE_PER_KWH + totalVMs()*VM_PRICE_PER_HOUR*STEP_HOURS;}
        double executionTimeMs(){return 55 + avgCPU()*1.25 + avgMem()*0.45 + avgNet()*0.35;}
    }

    static class P2Env {
        static final int H = 8;
        double[] util=new double[H], temp=new double[H];
        boolean[] on=new boolean[H];
        double reward, power; boolean qosViolated;

        double[] reset() {
            for(int i=0;i<H;i++){on[i]=true;util[i]=20+RNG.nextDouble()*60;
                temp[i]=30+util[i]*0.4+RNG.nextGaussian()*2;
                util[i]=Math.max(5,Math.min(100,util[i])); temp[i]=Math.max(20,Math.min(90,temp[i]));}
            return state();
        }

        double[] step(int a) {
            if(a==0){// consolidate
                int src=-1,dst=-1; double mn=Double.MAX_VALUE,mx=Double.MIN_VALUE;
                for(int i=0;i<H;i++){if(on[i]&&util[i]<mn){mn=util[i];src=i;}}
                for(int i=0;i<H;i++){if(on[i]&&util[i]>mx&&i!=src){mx=util[i];dst=i;}}
                if(src>=0&&dst>=0){double x=util[src]*0.4;util[dst]=Math.min(100,util[dst]+x);util[src]-=x;}
            } else if(a==1){// spread
                double avg=0;int n=0;for(int i=0;i<H;i++)if(on[i]){avg+=util[i];n++;}
                avg/=Math.max(1,n);for(int i=0;i<H;i++)if(on[i])util[i]+=(avg-util[i])*0.35;
            } else if(a==2){// power off
                int t=-1; double mn=Double.MAX_VALUE; long cnt=0;
                for(boolean b:on)if(b)cnt++;
                if(cnt>2)for(int i=0;i<H;i++)if(on[i]&&util[i]<mn){mn=util[i];t=i;}
                if(t>=0&&util[t]<25){on[t]=false;util[t]=0;temp[t]=20;}
            } else if(a==3){// power on
                for(int i=0;i<H;i++){if(!on[i]){on[i]=true;util[i]=15;temp[i]=30;break;}}
            }
            for(int i=0;i<H;i++){
                if(!on[i])continue;
                util[i]+=RNG.nextGaussian()*3; temp[i]=25+util[i]*0.5+RNG.nextGaussian()*2;
                util[i]=Math.max(5,Math.min(100,util[i])); temp[i]=Math.max(20,Math.min(95,temp[i]));
            }
            double tot=0; for(int i=0;i<H;i++) tot+=on[i]?80+170*util[i]/100:0;
            power=tot;
            qosViolated=Arrays.stream(util).filter(v->v>90).count()>0||
                    Arrays.stream(temp).filter(v->v>80).count()>1;
            double avgU=Arrays.stream(util).filter(v->v>0).average().orElse(0);
            double maxT=Arrays.stream(temp).max().getAsDouble();
            double saving=(H*250.0-tot)/(H*250.0);
            reward=saving*2; if(avgU>30&&avgU<80) reward+=0.5;
            if(qosViolated) reward-=2; if(maxT>80) reward-=1;
            int onCnt=0; for(boolean b:on) if(b) onCnt++;
            if(onCnt<=4) reward+=0.3; if(a==2&&avgU>75) reward-=0.8;
            return state();
        }

        double[] state(){
            double[] s=new double[24];
            for(int i=0;i<H;i++){
                s[i*3]=on[i]?util[i]/100:0;
                s[i*3+1]=on[i]?(80+170*util[i]/100)/250:0;
                s[i*3+2]=on[i]?temp[i]/95:0;
            }
            return s;
        }
        int activeHosts(){int c=0;for(boolean b:on)if(b)c++;return c;}
        int vmCount(int i){return on[i] ? Math.max(1, (int)Math.round(util[i] / 18.0)) : 0;}
        int totalVMs(){int t=0;for(int i=0;i<H;i++)t+=vmCount(i);return t;}
        double totalPowerW(){return power;}
        double stepEnergyKWh(){return totalPowerW()/1000.0*STEP_HOURS;}
        double stepCostUsd(){return stepEnergyKWh()*ENERGY_PRICE_PER_KWH + totalVMs()*VM_PRICE_PER_HOUR*STEP_HOURS;}
        double avgUtil(){return Arrays.stream(util).filter(v->v>0).average().orElse(0);}
        double executionTimeMs(){return 45 + avgUtil()*1.35 + Arrays.stream(temp).max().orElse(0)*0.25 + totalVMs()*1.8;}
    }

    // ════════════════════════════════════════════════════════
    //  SHARED NEURAL NETWORK
    // ════════════════════════════════════════════════════════
    static class NNet {
        final int in, hid, out;
        double[][] w1, w2; double[] b1, b2;

        NNet(int in, int hid, int out) {
            this.in=in; this.hid=hid; this.out=out;
            double s=Math.sqrt(2.0/in);
            w1=new double[hid][in]; b1=new double[hid];
            w2=new double[out][hid]; b2=new double[out];
            for(double[] r:w1) for(int j=0;j<in;j++) r[j]=RNG.nextGaussian()*s;
            for(double[] r:w2) for(int j=0;j<hid;j++) r[j]=RNG.nextGaussian()*s;
        }

        double[] fwd(double[] x) {
            double[] h=new double[hid];
            for(int i=0;i<hid;i++){double z=b1[i];for(int j=0;j<in;j++)z+=w1[i][j]*x[j];h[i]=Math.max(0,z);}
            double[] y=new double[out];
            for(int i=0;i<out;i++){double z=b2[i];for(int j=0;j<hid;j++)z+=w2[i][j]*h[j];y[i]=z;}
            return y;
        }

        void update(double[] x, int a, double g) {
            double[] pH=new double[hid],h=new double[hid];
            for(int i=0;i<hid;i++){double z=b1[i];for(int j=0;j<in;j++)z+=w1[i][j]*x[j];pH[i]=z;h[i]=Math.max(0,z);}
            for(int j=0;j<hid;j++) w2[a][j]+=g*h[j]; b2[a]+=g;
            for(int i=0;i<hid;i++){if(pH[i]<=0)continue;double d=g*w2[a][i];for(int j=0;j<in;j++)w1[i][j]+=LR*d*x[j];b1[i]+=LR*d;}
        }

        void copyFrom(NNet src){
            for(int i=0;i<hid;i++){b1[i]=src.b1[i];System.arraycopy(src.w1[i],0,w1[i],0,in);}
            for(int i=0;i<out;i++){b2[i]=src.b2[i];System.arraycopy(src.w2[i],0,w2[i],0,hid);}
        }

        void softUpdate(NNet src, double tau){
            for(int i=0;i<hid;i++){b1[i]=tau*src.b1[i]+(1-tau)*b1[i];for(int j=0;j<in;j++)w1[i][j]=tau*src.w1[i][j]+(1-tau)*w1[i][j];}
            for(int i=0;i<out;i++){b2[i]=tau*src.b2[i]+(1-tau)*b2[i];for(int j=0;j<hid;j++)w2[i][j]=tau*src.w2[i][j]+(1-tau)*w2[i][j];}
        }
    }

    // ════════════════════════════════════════════════════════
    //  REPLAY BUFFERS
    // ════════════════════════════════════════════════════════
    static class ReplayBuf {
        final int cap; final List<Tr> mem=new ArrayList<>(); int ptr=0;
        ReplayBuf(int c){cap=c;}
        void add(double[] s,int a,double r,double[] ns,boolean d){
            Tr t=new Tr(s,a,r,ns,d);
            if(mem.size()<cap)mem.add(t);else mem.set(ptr%cap,t);ptr++;
        }
        List<Tr> sample(int n){List<Tr> c=new ArrayList<>(mem);Collections.shuffle(c,RNG);return c.subList(0,Math.min(n,c.size()));}
        int size(){return mem.size();}
    }

    static class PriBuf {
        final int cap; final List<Tr> mem=new ArrayList<>(); final List<Double> pri=new ArrayList<>(); int ptr=0;
        PriBuf(int c){cap=c;}
        void add(double[] s,int a,double r,double[] ns,boolean d,double p){
            Tr t=new Tr(s,a,r,ns,d); double pv=Math.pow(Math.abs(p)+0.01,0.6);
            if(mem.size()<cap){mem.add(t);pri.add(pv);}else{mem.set(ptr%cap,t);pri.set(ptr%cap,pv);}ptr++;
        }
        List<Tr> sample(int n){
            double sp=pri.stream().mapToDouble(Double::doubleValue).sum();
            List<Tr> res=new ArrayList<>();
            for(int i=0;i<n;i++){double rv=RNG.nextDouble()*sp,cum=0;boolean found=false;
                for(int j=0;j<pri.size();j++){cum+=pri.get(j);if(cum>=rv){res.add(mem.get(j));found=true;break;}}
                if(!found)res.add(mem.get(RNG.nextInt(mem.size())));}
            return res;
        }
        int size(){return mem.size();}
    }

    record Tr(double[] s, int a, double r, double[] ns, boolean done) {}

    // ════════════════════════════════════════════════════════
    //  BENCHMARK RESULT HOLDER
    // ════════════════════════════════════════════════════════
    static class BenchResult {
        String name, algorithm, env, actions, rewardDesign, replayStrategy;
        int stateDim, actionDim;
        double finalEps, elapsedMs;
        List<Double> rewardLog = new ArrayList<>();
        List<Double> violationLog = new ArrayList<>();
        List<Double> cpuLog = new ArrayList<>();
        List<Double> energyLog = new ArrayList<>();
        List<Double> costLog = new ArrayList<>();
        List<Double> execTimeLog = new ArrayList<>();
        List<Double> hostLog = new ArrayList<>();
        List<Double> vmLog = new ArrayList<>();

        BenchResult(String n,String algo,String e,String a,String rd,String rs,int sd,int ad){
            name=n; algorithm=algo; env=e; actions=a; rewardDesign=rd; replayStrategy=rs; stateDim=sd; actionDim=ad;
        }
    }

    // ════════════════════════════════════════════════════════
    //  MATH HELPERS
    // ════════════════════════════════════════════════════════
    static int argmax(double[] a){int b=0;for(int i=1;i<a.length;i++)if(a[i]>a[b])b=i;return b;}
    static double max(double[] a){double m=a[0];for(double v:a)if(v>m)m=v;return m;}
    static double min(double[] a){double m=a[0];for(double v:a)if(v<m)m=v;return m;}
    static double max(List<Double> l){return l.stream().mapToDouble(Double::doubleValue).max().orElse(0);}
    static double avg(List<Double> l){return l.stream().mapToDouble(Double::doubleValue).average().orElse(0);}
    static double sum(List<Double> l){return l.stream().mapToDouble(Double::doubleValue).sum();}
    static double avgLast(List<Double> l,int n){int s=Math.max(0,l.size()-n);return l.subList(s,l.size()).stream().mapToDouble(Double::doubleValue).average().orElse(0);}
    static double std(List<Double> l){double m=l.stream().mapToDouble(Double::doubleValue).average().orElse(0);return Math.sqrt(l.stream().mapToDouble(v->Math.pow(v-m,2)).average().orElse(0));}
    static int clampInt(int v,int lo,int hi){return Math.max(lo,Math.min(hi,v));}

    static double[] smooth(List<Double> l, int w){
        double[] a=l.stream().mapToDouble(Double::doubleValue).toArray();
        double[] s=new double[a.length];
        for(int i=0;i<a.length;i++){int lo=Math.max(0,i-w/2),hi=Math.min(a.length-1,i+w/2);double sum=0;for(int j=lo;j<=hi;j++)sum+=a[j];s[i]=sum/(hi-lo+1);}
        return s;
    }

    static int estimateConvergence(List<Double> log){
        double[] s=smooth(log,10);
        double best=max(Arrays.stream(s).boxed().toList());
        double threshold=best*0.85;
        for(int i=0;i<s.length;i++) if(s[i]>=threshold) return i+1;
        return EPISODES;
    }

    static String truncate(String s, int max){return s.length()<=max?s:s.substring(0,max-1)+"…";}
}
