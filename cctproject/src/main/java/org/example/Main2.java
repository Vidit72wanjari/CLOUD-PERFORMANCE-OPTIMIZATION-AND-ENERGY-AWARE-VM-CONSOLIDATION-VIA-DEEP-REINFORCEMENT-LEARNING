package org.example;

import java.util.*;

/**
 * ============================================================
 *  Main2.java
 *  Problem: Energy-Aware VM Consolidation using Deep RL
 *  Algorithm: Double DQN (DDQN) with Prioritized Replay
 * ============================================================
 *
 *  HOW TO RUN IN IntelliJ:
 *  1. File → New Project → Java (no framework)
 *  2. src/ → New → Java Class → name it Main2
 *  3. Paste entire file → Run 'Main2.main()'
 *
 *  PROBLEM STATEMENT:
 *  Data centres waste ~40% energy on idle or lightly loaded VMs.
 *  Goal: An RL agent learns when to consolidate VMs onto fewer
 *  physical hosts, power off idle hosts, and migrate workloads —
 *  minimising total energy consumption while respecting QoS
 *  (response-time) SLAs.
 *
 *  DIFFERENCE FROM Main.java (Cloud Performance / DQN):
 *  ┌──────────────────┬──────────────────────┬──────────────────────┐
 *  │ Aspect           │ Main.java (Prob 1)    │ Main2.java (Prob 2)  │
 *  ├──────────────────┼──────────────────────┼──────────────────────┤
 *  │ Objective        │ Minimise SLA breach  │ Minimise energy use  │
 *  │ Algorithm        │ DQN                  │ Double DQN (DDQN)    │
 *  │ State            │ CPU/Mem/Net loads     │ VM count, host power │
 *  │                  │ per node (18-dim)     │ util, temp (24-dim)  │
 *  │ Actions          │ Scale/Migrate/Idle    │ Consolidate/Spread/  │
 *  │                  │ (4)                  │ PowerOff/PowerOn/Idle│
 *  │                  │                      │ (5)                  │
 *  │ Reward           │ CPU balance + SLA     │ Energy saved - QoS   │
 *  │ Replay           │ Uniform random        │ Prioritized (PER)    │
 *  │ Target update    │ Hard copy (ep 10)     │ Soft update (τ=0.01) │
 *  │ Hosts/Nodes      │ 6 nodes               │ 8 physical hosts     │
 *  └──────────────────┴──────────────────────┴──────────────────────┘
 * ============================================================
 */
public class Main2 {
    static final double STEP_HOURS = 1.0 / 60.0;
    static final double ENERGY_PRICE_KWH = 0.12;
    static final double VM_PRICE_PER_HOUR = 0.02;

    // ── Hyperparameters ──────────────────────────────────────
    static final int    NUM_HOSTS     = 8;
    static final int    MAX_VMS       = 6;       // max VMs per host
    static final int    STATE_DIM     = NUM_HOSTS * 3; // util, power, temp per host
    static final int    ACTION_DIM    = 5;        // consolidate,spread,poweroff,poweron,idle
    static final int    HIDDEN        = 128;
    static final int    BUFFER_SIZE   = 10_000;
    static final int    BATCH_SIZE    = 64;
    static final int    EPISODES      = 200;
    static final int    STEPS         = 50;
    static final double GAMMA         = 0.97;
    static final double LR            = 8e-4;
    static final double EPS_START     = 1.0;
    static final double EPS_MIN       = 0.05;
    static final double EPS_DECAY     = 0.994;
    static final double TAU           = 0.01;   // soft update rate
    static final double MAX_POWER_KW  = 250.0;  // max per-host power (W)
    static final double IDLE_POWER_KW = 80.0;   // idle host power (W)

    static final String[] ACTIONS = {
            "CONSOLIDATE", "SPREAD", "POWER-OFF", "POWER-ON", "IDLE"
    };
    static final Random RNG = new Random(7);

    // ════════════════════════════════════════════════════════
    public static void main(String[] args) {
        long startNs = System.nanoTime();
        printBanner();

        NeuralNet  online = new NeuralNet(STATE_DIM, HIDDEN, ACTION_DIM);
        NeuralNet  target = new NeuralNet(STATE_DIM, HIDDEN, ACTION_DIM);
        target.copyFrom(online);

        PrioritizedBuffer buffer = new PrioritizedBuffer(BUFFER_SIZE);
        EnergyEnv env = new EnergyEnv();

        double epsilon  = EPS_START;
        double bestAvg  = Double.NEGATIVE_INFINITY;
        int    qosBreaches = 0;
        List<Double> rewardLog   = new ArrayList<>();
        List<Double> energyLog   = new ArrayList<>();
        List<Double> costLog     = new ArrayList<>();
        List<Double> execLog     = new ArrayList<>();
        List<Integer> hostLog    = new ArrayList<>();
        List<Integer> vmLog      = new ArrayList<>();

        // ── Training loop ────────────────────────────────────
        for (int ep = 1; ep <= EPISODES; ep++) {

            double[] state     = env.reset();
            double   epReward  = 0;
            double   epEnergy  = 0;
            int      epQoS     = 0;

            for (int t = 0; t < STEPS; t++) {

                // 1. DDQN action selection: online net picks action,
                //    target net evaluates its value
                int action = (RNG.nextDouble() < epsilon)
                        ? RNG.nextInt(ACTION_DIM)
                        : ddqnAction(online, target, state);

                // 2. Environment step
                double[] next   = env.step(action);
                double   reward = env.lastReward();
                double   energy = env.lastEnergy();
                boolean  qos    = env.isQoSViolated();
                boolean  done   = (t == STEPS - 1);

                epReward += reward;
                epEnergy += energy;
                if (qos) epQoS++;

                // 3. Store with initial priority = max existing
                double tdErr = Math.abs(reward) + 0.01;
                buffer.add(state, action, reward, next, done, tdErr);
                state = next;

                // 4. Train with prioritized sampling
                if (buffer.size() >= BATCH_SIZE) {
                    trainDDQN(online, target, buffer);
                }

                // 5. Soft update target network every step
                target.softUpdate(online, TAU);
            }

            epsilon = Math.max(EPS_MIN, epsilon * EPS_DECAY);
            double avgR = epReward / STEPS;
            double avgE = epEnergy / STEPS;
            double avgCost = env.stepCostUsd();
            double avgExec = env.executionTimeMs();
            rewardLog.add(avgR);
            energyLog.add(avgE);
            costLog.add(avgCost);
            execLog.add(avgExec);
            hostLog.add(env.activeHosts());
            vmLog.add(env.totalVMs());
            qosBreaches += epQoS;
            if (avgR > bestAvg) bestAvg = avgR;

            if (ep == 1 || ep % 10 == 0) {
                double avg10 = avgLast(rewardLog, 10);
                System.out.printf(
                        "[Ep %3d/%d] reward=%+6.2f avg10=%+6.2f hosts=%-2d vms=%-2d exec=%6.1fms energy=%6.3fkWh cost=$%.4f qos=%-3d eps=%.3f %s%n",
                        ep, EPISODES, avgR, avg10,
                        env.activeHosts(), env.totalVMs(), avgExec,
                        avgE * STEP_HOURS / 1000.0, avgCost, epQoS,
                        epsilon, bar(ep, EPISODES, 20)
                );
            }
        }

        printSummary(rewardLog, energyLog, costLog, execLog, hostLog, vmLog,
                qosBreaches, bestAvg, (System.nanoTime() - startNs) / 1_000_000.0);
        printQTable(online);
        printPolicyDemo(online, env);
    }

    // ── DDQN: online net selects action, target net scores it ──
    static int ddqnAction(NeuralNet online, NeuralNet target, double[] s) {
        double[] qOnline = online.forward(s);
        double[] qTarget = target.forward(s);
        // Select action by online, evaluate by target (reduces overestimation)
        int best = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < ACTION_DIM; a++) {
            double combined = 0.6 * qOnline[a] + 0.4 * qTarget[a];
            if (combined > bestVal) { bestVal = combined; best = a; }
        }
        return best;
    }

    // ── DDQN training step ────────────────────────────────────
    static void trainDDQN(NeuralNet online, NeuralNet target, PrioritizedBuffer buf) {
        List<Transition> batch = buf.sample(BATCH_SIZE);
        for (Transition tr : batch) {
            double[] qOnline = online.forward(tr.s);
            // DDQN: next action chosen by online, value estimated by target
            int    nextA   = argmax(online.forward(tr.ns));
            double nextQ   = target.forward(tr.ns)[nextA];
            double tdTarget = tr.done ? tr.r : tr.r + GAMMA * nextQ;
            double error   = tdTarget - qOnline[tr.a];
            online.update(tr.s, tr.a, error * LR);
        }
    }

    // ════════════════════════════════════════════════════════
    //  ENERGY-AWARE CLOUD ENVIRONMENT
    // ════════════════════════════════════════════════════════
    static class EnergyEnv {
        double[] util    = new double[NUM_HOSTS]; // CPU utilization 0-100%
        int[]    vmCount = new int[NUM_HOSTS];     // VMs running on host
        boolean[]powered = new boolean[NUM_HOSTS]; // host powered on?
        double[] temp    = new double[NUM_HOSTS];  // temperature 20-90°C
        double lastReward, lastEnergy;
        boolean qosViolated;

        double[] reset() {
            for (int i = 0; i < NUM_HOSTS; i++) {
                powered[i] = true;
                vmCount[i] = 1 + RNG.nextInt(MAX_VMS);
                util[i]    = 10 + vmCount[i] * 12 + RNG.nextDouble() * 10;
                temp[i]    = 30 + util[i] * 0.4 + RNG.nextGaussian() * 3;
                util[i]    = clamp(util[i], 5, 100);
                temp[i]    = clamp(temp[i], 20, 90);
            }
            return state();
        }

        double[] step(int action) {
            switch (action) {
                case 0 -> consolidate();
                case 1 -> spread();
                case 2 -> powerOff();
                case 3 -> powerOn();
                case 4 -> { /* IDLE */ }
            }
            noise();

            double totalEnergy = totalPowerW();
            double avgUtil = avgUtil();
            qosViolated = Arrays.stream(util)
                    .filter((v) -> true)
                    .filter((v) -> v > 90).count() > 0 ||
                    Arrays.stream(temp).filter(v -> v > 80).count() > 1;

            lastEnergy = totalEnergy;
            lastReward = computeReward(action, totalEnergy, avgUtil);
            return state();
        }

        void consolidate() {
            // Move VMs from lightly loaded hosts to more loaded ones
            int src = -1, dst = -1;
            double minUtil = Double.MAX_VALUE;
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (powered[i] && util[i] < minUtil && vmCount[i] > 1) {
                    minUtil = util[i]; src = i;
                }
            }
            double maxUtil = Double.MIN_VALUE;
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (powered[i] && util[i] > maxUtil && i != src
                        && vmCount[i] < MAX_VMS) {
                    maxUtil = util[i]; dst = i;
                }
            }
            if (src >= 0 && dst >= 0 && vmCount[src] > 1) {
                int moved = Math.max(1, vmCount[src] / 2);
                vmCount[dst] += moved;
                vmCount[src] -= moved;
                util[dst] = clamp(util[dst] + moved * 14, 5, 100);
                util[src] = clamp(util[src] - moved * 13, 5, 100);
            }
        }

        void spread() {
            // Redistribute VMs evenly across all powered-on hosts
            int total = 0;
            int on = 0;
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (powered[i]) { total += vmCount[i]; on++; }
            }
            if (on > 0) {
                int base = total / on;
                for (int i = 0; i < NUM_HOSTS; i++) {
                    if (powered[i]) {
                        vmCount[i] = base;
                        util[i] = clamp(base * 14 + RNG.nextDouble() * 10, 5, 100);
                    }
                }
            }
        }

        void powerOff() {
            // Find most idle powered host and shut it down
            int target = -1;
            double minU = Double.MAX_VALUE;
            int onCount = 0; for (boolean b : powered) if (b) onCount++;
            if (onCount <= 2) return; // keep at least 2 hosts on
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (powered[i] && util[i] < minU) { minU = util[i]; target = i; }
            }
            if (target >= 0 && util[target] < 25) {
                powered[target] = false;
                // Migrate VMs to other hosts
                int migrate = vmCount[target];
                vmCount[target] = 0; util[target] = 0;
                for (int i = 0; i < NUM_HOSTS && migrate > 0; i++) {
                    if (powered[i] && vmCount[i] < MAX_VMS) {
                        int take = Math.min(migrate, MAX_VMS - vmCount[i]);
                        vmCount[i] += take;
                        util[i] = clamp(util[i] + take * 12, 5, 100);
                        migrate -= take;
                    }
                }
            }
        }

        void powerOn() {
            // Wake a powered-off host if load is high
            double avgU = avgUtil();
            if (avgU < 70) return;
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (!powered[i]) {
                    powered[i] = true;
                    vmCount[i] = 1;
                    util[i] = 15;
                    temp[i] = 32;
                    break;
                }
            }
        }

        void noise() {
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (!powered[i]) continue;
                util[i] += RNG.nextGaussian() * 3;
                temp[i] = 25 + util[i] * 0.5 + RNG.nextGaussian() * 2;
                util[i] = clamp(util[i], 5, 100);
                temp[i] = clamp(temp[i], 20, 95);
                vmCount[i] = Math.max(0, Math.min(MAX_VMS, vmCount[i]));
            }
        }

        double computeReward(int action, double totalPower, double avgUtil) {
            double maxPossible = NUM_HOSTS * MAX_POWER_KW;
            double energySaving = (maxPossible - totalPower) / maxPossible; // 0-1

            double r = 0;
            r += energySaving * 2.0;          // reward energy efficiency
            if (avgUtil > 30 && avgUtil < 80)  r += 0.5; // balanced load bonus
            if (qosViolated)                   r -= 2.0; // QoS penalty
            if (activeHosts() <= 4)            r += 0.3; // consolidation bonus
            double maxT = Arrays.stream(temp).max().getAsDouble();
            if (maxT > 80)                     r -= 1.0; // thermal penalty
            if (action == 2 && avgUtil > 75)   r -= 0.8; // wrong power-off
            if (action == 3 && avgUtil < 40)   r -= 0.5; // unnecessary power-on
            return r;
        }

        double[] state() {
            double[] s = new double[STATE_DIM];
            for (int i = 0; i < NUM_HOSTS; i++) {
                s[i * 3]     = powered[i] ? util[i] / 100.0 : 0.0;
                s[i * 3 + 1] = powered[i] ? powerW(i) / MAX_POWER_KW : 0.0;
                s[i * 3 + 2] = powered[i] ? temp[i]  / 95.0 : 0.0;
            }
            return s;
        }

        double powerW(int i) {
            if (!powered[i]) return 0;
            return IDLE_POWER_KW + (MAX_POWER_KW - IDLE_POWER_KW) * util[i] / 100.0;
        }

        double totalPowerW() {
            double t = 0;
            for (int i = 0; i < NUM_HOSTS; i++) t += powerW(i);
            return t;
        }

        double stepEnergyKWh() {
            return totalPowerW() / 1000.0 * STEP_HOURS;
        }

        double stepCostUsd() {
            return stepEnergyKWh() * ENERGY_PRICE_KWH + totalVMs() * VM_PRICE_PER_HOUR * STEP_HOURS;
        }

        double avgUtil() {
            int on = 0; double sum = 0;
            for (int i = 0; i < NUM_HOSTS; i++) {
                if (powered[i]) { sum += util[i]; on++; }
            }
            return on == 0 ? 0 : sum / on;
        }

        int activeHosts() {
            int c = 0;
            for (boolean p : powered) if (p) c++;
            return c;
        }

        int totalVMs() {
            int total = 0;
            for (int count : vmCount) total += count;
            return total;
        }

        double executionTimeMs() {
            double hottest = Arrays.stream(temp).max().orElse(0);
            return 45.0 + avgUtil() * 1.35 + hottest * 0.25 + totalVMs() * 1.8;
        }

        boolean isQoSViolated() { return qosViolated; }
        double  lastReward()    { return lastReward; }
        double  lastEnergy()    { return lastEnergy; }

        String stateName() {
            double avg = avgUtil();
            double pw  = totalPowerW();
            if (pw > 1600)  return "HIGH-ENERGY";
            if (avg > 80)   return "OVERLOADED";
            if (avg > 40)   return "EFFICIENT";
            return "UNDER-USED";
        }
    }

    // ════════════════════════════════════════════════════════
    //  NEURAL NETWORK (same 2-layer MLP, reused from Main.java)
    // ════════════════════════════════════════════════════════
    static class NeuralNet {
        final int in, hid, out;
        double[][] w1, w2;
        double[]   b1, b2;

        NeuralNet(int in, int hid, int out) {
            this.in = in; this.hid = hid; this.out = out;
            w1 = xavier(hid, in); b1 = new double[hid];
            w2 = xavier(out, hid); b2 = new double[out];
        }

        double[][] xavier(int r, int c) {
            double s = Math.sqrt(2.0 / c);
            double[][] m = new double[r][c];
            for (double[] row : m)
                for (int j = 0; j < c; j++) row[j] = RNG.nextGaussian() * s;
            return m;
        }

        double[] forward(double[] x) {
            double[] h = new double[hid];
            for (int i = 0; i < hid; i++) {
                double z = b1[i];
                for (int j = 0; j < in; j++) z += w1[i][j] * x[j];
                h[i] = Math.max(0, z);
            }
            double[] y = new double[out];
            for (int i = 0; i < out; i++) {
                double z = b2[i];
                for (int j = 0; j < hid; j++) z += w2[i][j] * h[j];
                y[i] = z;
            }
            return y;
        }

        void update(double[] x, int a, double grad) {
            double[] preH = new double[hid], h = new double[hid];
            for (int i = 0; i < hid; i++) {
                double z = b1[i];
                for (int j = 0; j < in; j++) z += w1[i][j] * x[j];
                preH[i] = z; h[i] = Math.max(0, z);
            }
            for (int j = 0; j < hid; j++) w2[a][j] += grad * h[j];
            b2[a] += grad;
            for (int i = 0; i < hid; i++) {
                if (preH[i] <= 0) continue;
                double d = grad * w2[a][i];
                for (int j = 0; j < in; j++) w1[i][j] += LR * d * x[j];
                b1[i] += LR * d;
            }
        }

        void copyFrom(NeuralNet src) {
            for (int i = 0; i < hid; i++) {
                b1[i] = src.b1[i];
                System.arraycopy(src.w1[i], 0, w1[i], 0, in);
            }
            for (int i = 0; i < out; i++) {
                b2[i] = src.b2[i];
                System.arraycopy(src.w2[i], 0, w2[i], 0, hid);
            }
        }

        // Soft update: θ_target = τ·θ_online + (1−τ)·θ_target
        void softUpdate(NeuralNet src, double tau) {
            for (int i = 0; i < hid; i++) {
                b1[i] = tau * src.b1[i] + (1 - tau) * b1[i];
                for (int j = 0; j < in; j++)
                    w1[i][j] = tau * src.w1[i][j] + (1 - tau) * w1[i][j];
            }
            for (int i = 0; i < out; i++) {
                b2[i] = tau * src.b2[i] + (1 - tau) * b2[i];
                for (int j = 0; j < hid; j++)
                    w2[i][j] = tau * src.w2[i][j] + (1 - tau) * w2[i][j];
            }
        }
    }

    // ════════════════════════════════════════════════════════
    //  PRIORITIZED EXPERIENCE REPLAY
    //  Transitions with higher TD error are sampled more often
    // ════════════════════════════════════════════════════════
    static class PrioritizedBuffer {
        final int cap;
        final List<Transition> mem = new ArrayList<>();
        final List<Double> priorities = new ArrayList<>();
        int ptr = 0;
        static final double ALPHA = 0.6; // priority exponent

        PrioritizedBuffer(int cap) { this.cap = cap; }

        void add(double[] s, int a, double r, double[] ns, boolean done, double priority) {
            Transition t = new Transition(s, a, r, ns, done);
            double p = Math.pow(Math.abs(priority) + 0.01, ALPHA);
            if (mem.size() < cap) { mem.add(t); priorities.add(p); }
            else {
                mem.set(ptr % cap, t);
                priorities.set(ptr % cap, p);
            }
            ptr++;
        }

        List<Transition> sample(int n) {
            double sumP = priorities.stream().mapToDouble(Double::doubleValue).sum();
            List<Integer> chosen = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                double r = RNG.nextDouble() * sumP;
                double cum = 0;
                for (int j = 0; j < priorities.size(); j++) {
                    cum += priorities.get(j);
                    if (cum >= r) { chosen.add(j); break; }
                }
                if (chosen.size() < i + 1) chosen.add(RNG.nextInt(mem.size()));
            }
            List<Transition> result = new ArrayList<>();
            for (int idx : chosen) result.add(mem.get(idx));
            return result;
        }

        int size() { return mem.size(); }
    }

    record Transition(double[] s, int a, double r, double[] ns, boolean done) {}

    // ════════════════════════════════════════════════════════
    //  PRINT HELPERS
    // ════════════════════════════════════════════════════════
    static void printBanner() {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     ENERGY-AWARE VM CONSOLIDATION - DDQN AGENT          ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.printf ("║  Hosts: %-3d  State: %-3d  Actions: %-3d  Episodes: %-5d  ║%n",
                NUM_HOSTS, STATE_DIM, ACTION_DIM, EPISODES);
        System.out.printf ("║  γ=%.2f  α=%.4f  ε: %.2f→%.2f  τ(soft)=%.3f       ║%n",
                GAMMA, LR, EPS_START, EPS_MIN, TAU);
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    static void printSummary(List<Double> rlog, List<Double> elog, List<Double> clog,
                             List<Double> xlog, List<Integer> hlog, List<Integer> vlog,
                             int qos, double best, double runtimeMs) {
        double avgR = rlog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgE = elog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double totalE = elog.stream().mapToDouble(Double::doubleValue).sum() * STEP_HOURS / 1000.0;
        double avgC = clog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double totalC = clog.stream().mapToDouble(Double::doubleValue).sum();
        double avgX = xlog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgH = hlog.stream().mapToInt(Integer::intValue).average().orElse(0);
        double avgV = vlog.stream().mapToInt(Integer::intValue).average().orElse(0);
        double avg10= avgLast(rlog, 10);
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║                   TRAINING COMPLETE                      ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.printf ("║  Episodes          : %-5d                              ║%n", rlog.size());
        System.out.printf ("║  Avg reward        : %+.3f                             ║%n", avgR);
        System.out.printf ("║  Best episode      : %+.3f                             ║%n", best);
        System.out.printf ("║  Final avg10 reward: %+.3f                             ║%n", avg10);
        System.out.printf ("║  Avg energy (W)    : %-6.1f                             ║%n", avgE);
        System.out.printf ("║  Total energy      : %-6.4f kWh                         ║%n", totalE);
        System.out.printf ("║  Avg cost / ep     : $%-7.4f                            ║%n", avgC);
        System.out.printf ("║  Total cost        : $%-7.4f                            ║%n", totalC);
        System.out.printf ("║  Avg exec time     : %-6.2f ms                          ║%n", avgX);
        System.out.printf ("║  Avg hosts / VMs   : %-4.1f / %-6.1f                     ║%n", avgH, avgV);
        System.out.printf ("║  Total QoS breaches: %-5d                              ║%n", qos);
        System.out.printf ("║  Training runtime  : %-8.2f ms                         ║%n", runtimeMs);
        System.out.println("╚══════════════════════════════════════════════════════════╝");
    }

    static void printQTable(NeuralNet q) {
        System.out.println();
        System.out.println("Q-VALUE TABLE  (greedy action shown in [ ])");
        System.out.println("─".repeat(72));
        System.out.printf("%-14s %13s %8s %10s %10s %8s%n",
                "State", "CONSOLIDATE", "SPREAD", "POWER-OFF", "POWER-ON", "IDLE");
        System.out.println("─".repeat(72));

        double[][] samples = {
                stateVec(85, 75, 70, false),  // overloaded
                stateVec(55, 50, 45, false),  // efficient
                stateVec(15, 10, 35, false),  // under-used
                stateVec(92, 88, 80, false),  // high energy
        };
        String[] names = { "OVERLOADED", "EFFICIENT", "UNDER-USED", "HIGH-ENERGY" };

        for (int i = 0; i < samples.length; i++) {
            double[] qv  = q.forward(samples[i]);
            int      best = argmax(qv);
            System.out.printf("%-14s", names[i]);
            for (int a = 0; a < ACTION_DIM; a++) {
                String cell = String.format("%+.3f", qv[a]);
                System.out.printf("%13s", a == best ? "[" + cell + "]" : cell);
            }
            System.out.println();
        }
        System.out.println("─".repeat(72));
    }

    static void printPolicyDemo(NeuralNet q, EnergyEnv env) {
        System.out.println();
        System.out.println("POLICY DEMO  (10 greedy steps, ε = 0)");
        System.out.println("─".repeat(66));
        System.out.printf("%-6s %-12s %-14s %-9s %-8s %-6s %-6s %-8s%n",
                "Step", "State", "Action", "Reward", "Power(W)", "Hosts", "VMs", "ExecMs");
        System.out.println("─".repeat(66));

        double[] state = env.reset();
        double total = 0;
        for (int t = 1; t <= 10; t++) {
            int action = ddqnAction(q, q, state);
            state = env.step(action);
            total += env.lastReward();
            System.out.printf("%-6d %-12s %-14s %+6.2f   %7.1f   %-6d %-6d %-8.1f%n",
                    t, env.stateName(), ACTIONS[action],
                    env.lastReward(), env.lastEnergy(), env.activeHosts(),
                    env.totalVMs(), env.executionTimeMs());
        }
        System.out.println("─".repeat(66));
        System.out.printf("Total reward (10 steps): %+.2f%n", total);
    }

    // ── Utilities ────────────────────────────────────────────
    static int argmax(double[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
    static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
    static double avgLast(List<Double> list, int n) {
        int start = Math.max(0, list.size() - n);
        return list.subList(start, list.size())
                .stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }
    static double[] stateVec(double util, double pw, double temp, boolean off) {
        double[] s = new double[STATE_DIM];
        for (int i = 0; i < NUM_HOSTS; i++) {
            s[i*3]   = off ? 0 : util/100.0 + RNG.nextGaussian()*0.01;
            s[i*3+1] = off ? 0 : pw/250.0   + RNG.nextGaussian()*0.01;
            s[i*3+2] = off ? 0 : temp/95.0  + RNG.nextGaussian()*0.01;
        }
        return s;
    }
    static String bar(int cur, int tot, int w) {
        int f = (int)((double)cur/tot*w);
        return "[" + "█".repeat(f) + "░".repeat(w-f) + "]";
    }
}
