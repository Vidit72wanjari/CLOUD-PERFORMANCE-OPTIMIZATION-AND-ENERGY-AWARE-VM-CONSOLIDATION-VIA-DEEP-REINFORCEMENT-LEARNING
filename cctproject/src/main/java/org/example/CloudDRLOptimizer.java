package org.example;

import java.util.*;

/**
 * ============================================================
 *  Cloud Performance Optimization using Deep Reinforcement Learning
 *  Simulated DQN Agent — runs in IntelliJ with zero dependencies
 * ============================================================
 *
 *  HOW TO RUN IN IntelliJ:
 *  1. File → New Project → Java (no framework needed)
 *  2. Create a new file: src/CloudDRLOptimizer.java
 *  3. Paste this entire file
 *  4. Right-click → Run 'CloudDRLOptimizer.main()'
 * ============================================================
 */
public class CloudDRLOptimizer {
    static final double STEP_HOURS = 1.0 / 60.0;
    static final double ENERGY_PRICE_PER_KWH = 0.12;
    static final double VM_PRICE_PER_HOUR = 0.018;
    static final double HOST_IDLE_POWER_W = 90.0;
    static final double HOST_MAX_POWER_W = 220.0;

    // ── Hyperparameters ─────────────────────────────────────
    static final int    NUM_NODES      = 6;
    static final int    STATE_DIM      = NUM_NODES * 3;  // cpu, mem, net per node
    static final int    ACTION_DIM     = 4;              // scale-up, scale-down, migrate, idle
    static final int    HIDDEN_SIZE    = 128;
    static final int    REPLAY_BUFFER  = 10_000;
    static final int    BATCH_SIZE     = 64;
    static final int    EPISODES       = 200;
    static final int    STEPS_PER_EP   = 50;
    static final double GAMMA          = 0.95;
    static final double LR             = 1e-3;
    static final double EPSILON_START  = 1.0;
    static final double EPSILON_MIN    = 0.05;
    static final double EPSILON_DECAY  = 0.995;
    static final int    TARGET_UPDATE  = 10;  // sync target network every N episodes

    // ── Action labels ────────────────────────────────────────
    static final String[] ACTIONS = {"SCALE-UP", "SCALE-DOWN", "MIGRATE", "IDLE"};
    static final String[] STATES  = {"OVERLOADED", "BALANCED", "UNDERUTIL", "CRITICAL"};

    static final Random RNG = new Random(42);

    // ── Entry point ──────────────────────────────────────────
    public static void main(String[] args) {
        long startNs = System.nanoTime();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║      CLOUD PERFORMANCE OPTIMIZATION - DQN AGENT         ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.printf ("║  Nodes: %-4d  State dim: %-4d  Action dim: %-4d         ║%n",
                NUM_NODES, STATE_DIM, ACTION_DIM);
        System.out.printf ("║  γ=%.2f  α=%.4f  ε: %.2f→%.2f  Buffer: %-6d      ║%n",
                GAMMA, LR, EPSILON_START, EPSILON_MIN, REPLAY_BUFFER);
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();

        DQNAgent agent = new DQNAgent();
        CloudEnv env   = new CloudEnv();
        TrainingStats stats = new TrainingStats();

        // ── Training loop ────────────────────────────────────
        for (int ep = 1; ep <= EPISODES; ep++) {

            double[] state = env.reset();
            double epReward = 0;
            int slaBreaches = 0;

            for (int step = 0; step < STEPS_PER_EP; step++) {
                int action = agent.selectAction(state);
                StepResult sr = env.step(action);
                agent.storeTransition(state, action, sr.reward, sr.nextState, sr.done);
                agent.train();
                state = sr.nextState;
                epReward += sr.reward;
                if (sr.slaViolated) slaBreaches++;
                if (sr.done) break;
            }

            stats.record(epReward / STEPS_PER_EP, slaBreaches, env);
            agent.decayEpsilon();

            if (ep % TARGET_UPDATE == 0) {
                agent.syncTargetNetwork();
            }

            // Print progress every 10 episodes
            if (ep % 10 == 0 || ep == 1) {
                printEpisodeReport(ep, stats, env, agent);
            }
        }

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║                   TRAINING COMPLETE                      ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        stats.printFinal((System.nanoTime() - startNs) / 1_000_000.0);
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        System.out.println();
        printQTable(agent);
        System.out.println();
        printPolicy(agent, env);
    }

    // ── Episode report ───────────────────────────────────────
    static void printEpisodeReport(int ep, TrainingStats stats, CloudEnv env, DQNAgent agent) {
        double avg10 = stats.avgRewardLast(10);
        System.out.printf(
                "[Ep %3d/%d] reward=%-6.2f avg10=%-6.2f cpu=%-4.1f%% hosts=%d vms=%-2d exec=%6.1fms energy=%5.3fkWh cost=$%.4f sla=%-3d ε=%.3f %s%n",
                ep, EPISODES,
                stats.lastReward(),
                avg10,
                env.avgCPU(),
                env.activeHosts(),
                env.totalVMs(),
                env.executionTimeMs(),
                env.stepEnergyKWh(),
                env.stepCostUsd(),
                stats.lastSLA(),
                agent.epsilon,
                progressBar(ep, EPISODES, 20)
        );
    }

    static String progressBar(int current, int total, int width) {
        int filled = (int)((double)current / total * width);
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < width; i++) sb.append(i < filled ? "█" : "░");
        sb.append("]");
        return sb.toString();
    }

    // ── Q-table snapshot ─────────────────────────────────────
    static void printQTable(DQNAgent agent) {
        System.out.println("Q-VALUE SNAPSHOT (sampled state vectors → action values)");
        System.out.println("─".repeat(62));
        System.out.printf("%-14s %12s %13s %10s %10s%n",
                "State", "SCALE-UP", "SCALE-DOWN", "MIGRATE", "IDLE");
        System.out.println("─".repeat(62));

        double[][] sampleStates = {
                buildState(90, 80, 70), // overloaded
                buildState(55, 50, 45), // balanced
                buildState(20, 15, 10), // underutil
                buildState(95, 95, 90), // critical
        };
        String[] labels = {"OVERLOADED", "BALANCED", "UNDERUTIL", "CRITICAL"};

        for (int i = 0; i < sampleStates.length; i++) {
            double[] q = agent.qNetwork.forward(sampleStates[i]);
            int best = argmax(q);
            System.out.printf("%-14s", labels[i]);
            for (int a = 0; a < ACTION_DIM; a++) {
                String cell = String.format("%+.3f", q[a]);
                if (a == best) cell = "[" + cell + "]";
                System.out.printf("%13s", cell);
            }
            System.out.println();
        }
        System.out.println("─".repeat(62));
        System.out.println("  [ ] = greedy action selected by policy");
    }

    // ── Policy demo ──────────────────────────────────────────
    static void printPolicy(DQNAgent agent, CloudEnv env) {
        System.out.println("LEARNED POLICY DEMO (10 steps, ε=0 greedy)");
        System.out.println("─".repeat(62));
        System.out.printf("%-6s %-12s %-12s %-8s %-8s %-6s %-6s %-8s%n",
                "Step", "State", "Action", "Reward", "CPU%", "Hosts", "VMs", "ExecMs");
        System.out.println("─".repeat(62));

        double[] state = env.reset();
        double totalR = 0;
        for (int t = 1; t <= 10; t++) {
            int action = agent.greedyAction(state);
            StepResult sr = env.step(action);
            totalR += sr.reward;
            System.out.printf("%-6d %-12s %-12s %-8.2f %-8.1f %-6d %-6d %-8.1f%n",
                    t, env.currentStateName(), ACTIONS[action], sr.reward, env.avgCPU(),
                    env.activeHosts(), env.totalVMs(), env.executionTimeMs());
            state = sr.nextState;
        }
        System.out.println("─".repeat(62));
        System.out.printf("Total reward over 10 steps: %.2f%n", totalR);
    }

    static double[] buildState(double cpu, double mem, double net) {
        double[] s = new double[STATE_DIM];
        for (int i = 0; i < NUM_NODES; i++) {
            s[i * 3]     = cpu / 100.0 + RNG.nextGaussian() * 0.02;
            s[i * 3 + 1] = mem / 100.0 + RNG.nextGaussian() * 0.02;
            s[i * 3 + 2] = net / 100.0 + RNG.nextGaussian() * 0.02;
        }
        return s;
    }

    static int argmax(double[] arr) {
        int best = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }

    // ════════════════════════════════════════════════════════
    //  CLOUD ENVIRONMENT
    // ════════════════════════════════════════════════════════
    static class CloudEnv {
        double[] cpuLoads  = new double[NUM_NODES];
        double[] memLoads  = new double[NUM_NODES];
        double[] netLoads  = new double[NUM_NODES];
        int stepCount = 0;

        double[] reset() {
            for (int i = 0; i < NUM_NODES; i++) {
                cpuLoads[i] = 30 + RNG.nextDouble() * 60;
                memLoads[i] = 20 + RNG.nextDouble() * 70;
                netLoads[i] = 10 + RNG.nextDouble() * 50;
            }
            stepCount = 0;
            return getState();
        }

        StepResult step(int action) {
            applyAction(action);
            addWorkloadNoise();
            stepCount++;

            double reward = computeReward(action);
            boolean slaViolated = Arrays.stream(cpuLoads).anyMatch(v -> v > 90);
            boolean done = stepCount >= STEPS_PER_EP;

            return new StepResult(getState(), reward, done, slaViolated);
        }

        void applyAction(int action) {
            switch (action) {
                case 0: // SCALE-UP: redistribute load to most idle node
                    int maxNode = 0;
                    for (int i = 1; i < NUM_NODES; i++)
                        if (cpuLoads[i] > cpuLoads[maxNode]) maxNode = i;
                    int minNode = 0;
                    for (int i = 1; i < NUM_NODES; i++)
                        if (cpuLoads[i] < cpuLoads[minNode]) minNode = i;
                    double xfer = (cpuLoads[maxNode] - cpuLoads[minNode]) * 0.3;
                    cpuLoads[maxNode] -= xfer;
                    cpuLoads[minNode] += xfer * 0.7;
                    break;

                case 1: // SCALE-DOWN: consolidate to fewer nodes
                    for (int i = 0; i < NUM_NODES; i++)
                        cpuLoads[i] = Math.min(100, cpuLoads[i] * 1.1);
                    break;

                case 2: // MIGRATE: rebalance all nodes toward mean
                    double mean = avgCPU();
                    for (int i = 0; i < NUM_NODES; i++)
                        cpuLoads[i] += (mean - cpuLoads[i]) * 0.4;
                    break;

                case 3: // IDLE: no action
                    break;
            }
            // Clamp
            for (int i = 0; i < NUM_NODES; i++) {
                cpuLoads[i] = Math.max(5, Math.min(100, cpuLoads[i]));
                memLoads[i] = Math.max(5, Math.min(100, memLoads[i]));
            }
        }

        void addWorkloadNoise() {
            for (int i = 0; i < NUM_NODES; i++) {
                cpuLoads[i] += RNG.nextGaussian() * 3;
                memLoads[i] += RNG.nextGaussian() * 2;
                netLoads[i] += RNG.nextGaussian() * 4;
                cpuLoads[i] = Math.max(5, Math.min(100, cpuLoads[i]));
                memLoads[i] = Math.max(5, Math.min(100, memLoads[i]));
                netLoads[i] = Math.max(5, Math.min(100, netLoads[i]));
            }
        }

        double computeReward(int action) {
            double avg = avgCPU();
            double max = Arrays.stream(cpuLoads).max().getAsDouble();
            double spread = max - Arrays.stream(cpuLoads).min().getAsDouble();

            double r = 0;
            if (avg >= 40 && avg <= 70) r += 1.0;   // balanced utilization
            if (spread < 20)            r += 0.5;   // even distribution
            if (max > 90)               r -= 2.5;   // SLA violation
            if (max > 85)               r -= 1.0;   // near-SLA warning
            if (action == 1 && avg < 30) r -= 0.5;  // unnecessary scale-down
            if (action == 0 && avg > 75) r += 0.3;  // smart scale-up
            if (action == 3 && avg > 80) r -= 1.0;  // idle when should act
            return r;
        }

        double[] getState() {
            double[] s = new double[STATE_DIM];
            for (int i = 0; i < NUM_NODES; i++) {
                s[i * 3]     = cpuLoads[i] / 100.0;
                s[i * 3 + 1] = memLoads[i] / 100.0;
                s[i * 3 + 2] = netLoads[i] / 100.0;
            }
            return s;
        }

        double avgCPU() {
            return Arrays.stream(cpuLoads).average().orElse(0);
        }

        double avgMem() {
            return Arrays.stream(memLoads).average().orElse(0);
        }

        double avgNet() {
            return Arrays.stream(netLoads).average().orElse(0);
        }

        int activeHosts() {
            return NUM_NODES;
        }

        int vmCountAt(int index) {
            return Math.max(1, (int)Math.round((cpuLoads[index] + memLoads[index]) / 35.0));
        }

        int totalVMs() {
            int total = 0;
            for (int i = 0; i < NUM_NODES; i++) total += vmCountAt(i);
            return total;
        }

        double hostPowerW(int index) {
            double weightedUtil = (cpuLoads[index] * 0.6 + memLoads[index] * 0.25 + netLoads[index] * 0.15) / 100.0;
            return HOST_IDLE_POWER_W + (HOST_MAX_POWER_W - HOST_IDLE_POWER_W) * weightedUtil;
        }

        double totalPowerW() {
            double total = 0;
            for (int i = 0; i < NUM_NODES; i++) total += hostPowerW(i);
            return total;
        }

        double stepEnergyKWh() {
            return totalPowerW() / 1000.0 * STEP_HOURS;
        }

        double stepCostUsd() {
            return stepEnergyKWh() * ENERGY_PRICE_PER_KWH + totalVMs() * VM_PRICE_PER_HOUR * STEP_HOURS;
        }

        double executionTimeMs() {
            return 55.0 + avgCPU() * 1.25 + avgMem() * 0.45 + avgNet() * 0.35;
        }

        String currentStateName() {
            double avg = avgCPU();
            double max = Arrays.stream(cpuLoads).max().getAsDouble();
            if (max > 90) return "CRITICAL";
            if (avg > 70) return "OVERLOADED";
            if (avg > 40) return "BALANCED";
            return "UNDERUTIL";
        }
    }

    static class StepResult {
        double[] nextState;
        double reward;
        boolean done, slaViolated;
        StepResult(double[] ns, double r, boolean d, boolean sla) {
            nextState = ns; reward = r; done = d; slaViolated = sla;
        }
    }

    // ════════════════════════════════════════════════════════
    //  DQN AGENT
    // ════════════════════════════════════════════════════════
    static class DQNAgent {
        NeuralNetwork qNetwork;
        NeuralNetwork targetNetwork;
        ReplayBuffer  buffer;
        double epsilon = EPSILON_START;

        DQNAgent() {
            qNetwork      = new NeuralNetwork(STATE_DIM, HIDDEN_SIZE, ACTION_DIM);
            targetNetwork = new NeuralNetwork(STATE_DIM, HIDDEN_SIZE, ACTION_DIM);
            syncTargetNetwork();
            buffer = new ReplayBuffer(REPLAY_BUFFER);
        }

        int selectAction(double[] state) {
            if (RNG.nextDouble() < epsilon) return RNG.nextInt(ACTION_DIM);
            return greedyAction(state);
        }

        int greedyAction(double[] state) {
            double[] q = qNetwork.forward(state);
            return argmax(q);
        }

        void storeTransition(double[] s, int a, double r, double[] ns, boolean done) {
            buffer.add(new Transition(s, a, r, ns, done));
        }

        void train() {
            if (buffer.size() < BATCH_SIZE) return;
            List<Transition> batch = buffer.sample(BATCH_SIZE);
            for (Transition t : batch) {
                double[] qVals   = qNetwork.forward(t.state);
                double[] qNext   = targetNetwork.forward(t.nextState);
                double target    = t.done ? t.reward : t.reward + GAMMA * max(qNext);
                double tdError   = target - qVals[t.action];
                // Simplified gradient step — in-place weight nudge
                qNetwork.updateWeights(t.state, t.action, tdError * LR);
            }
        }

        void decayEpsilon() {
            epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
        }

        void syncTargetNetwork() {
            targetNetwork.copyWeightsFrom(qNetwork);
        }

        static double max(double[] arr) {
            double m = arr[0];
            for (double v : arr) if (v > m) m = v;
            return m;
        }
    }

    // ════════════════════════════════════════════════════════
    //  NEURAL NETWORK (2-layer MLP, ReLU hidden, linear out)
    // ════════════════════════════════════════════════════════
    static class NeuralNetwork {
        int in, hidden, out;
        double[][] w1, w2;
        double[]   b1, b2;

        NeuralNetwork(int in, int hidden, int out) {
            this.in = in; this.hidden = hidden; this.out = out;
            w1 = xavier(hidden, in);
            b1 = new double[hidden];
            w2 = xavier(out, hidden);
            b2 = new double[out];
        }

        double[][] xavier(int rows, int cols) {
            double scale = Math.sqrt(2.0 / cols);
            double[][] m = new double[rows][cols];
            for (double[] row : m)
                for (int j = 0; j < cols; j++) row[j] = RNG.nextGaussian() * scale;
            return m;
        }

        double[] forward(double[] x) {
            // Layer 1 + ReLU
            double[] h = new double[hidden];
            for (int i = 0; i < hidden; i++) {
                double sum = b1[i];
                for (int j = 0; j < in; j++) sum += w1[i][j] * x[j];
                h[i] = Math.max(0, sum); // ReLU
            }
            // Layer 2 (linear output)
            double[] y = new double[out];
            for (int i = 0; i < out; i++) {
                double sum = b2[i];
                for (int j = 0; j < hidden; j++) sum += w2[i][j] * h[j];
                y[i] = sum;
            }
            return y;
        }

        /** Simplified single-step gradient update for action `a` */
        void updateWeights(double[] x, int a, double gradStep) {
            double[] h = new double[hidden];
            double[] preH = new double[hidden];
            for (int i = 0; i < hidden; i++) {
                double sum = b1[i];
                for (int j = 0; j < in; j++) sum += w1[i][j] * x[j];
                preH[i] = sum;
                h[i] = Math.max(0, sum);
            }
            // Backprop through w2 for action a
            for (int j = 0; j < hidden; j++) {
                w2[a][j] += gradStep * h[j];
            }
            b2[a] += gradStep;
            // Backprop through w1
            for (int i = 0; i < hidden; i++) {
                if (preH[i] <= 0) continue; // ReLU gate
                double delta = gradStep * w2[a][i];
                for (int j = 0; j < in; j++) {
                    w1[i][j] += LR * delta * x[j];
                }
                b1[i] += LR * delta;
            }
        }

        void copyWeightsFrom(NeuralNetwork src) {
            for (int i = 0; i < hidden; i++) {
                b1[i] = src.b1[i];
                System.arraycopy(src.w1[i], 0, w1[i], 0, in);
            }
            for (int i = 0; i < out; i++) {
                b2[i] = src.b2[i];
                System.arraycopy(src.w2[i], 0, w2[i], 0, hidden);
            }
        }
    }

    // ════════════════════════════════════════════════════════
    //  REPLAY BUFFER (experience replay)
    // ════════════════════════════════════════════════════════
    static class ReplayBuffer {
        final int capacity;
        final List<Transition> buffer = new ArrayList<>();
        int ptr = 0;

        ReplayBuffer(int capacity) { this.capacity = capacity; }

        void add(Transition t) {
            if (buffer.size() < capacity) buffer.add(t);
            else buffer.set(ptr % capacity, t);
            ptr++;
        }

        List<Transition> sample(int n) {
            List<Transition> copy = new ArrayList<>(buffer);
            Collections.shuffle(copy, RNG);
            return copy.subList(0, Math.min(n, copy.size()));
        }

        int size() { return buffer.size(); }
    }

    static class Transition {
        double[] state, nextState;
        int action;
        double reward;
        boolean done;
        Transition(double[] s, int a, double r, double[] ns, boolean d) {
            state = s; action = a; reward = r; nextState = ns; done = d;
        }
    }

    // ════════════════════════════════════════════════════════
    //  TRAINING STATISTICS
    // ════════════════════════════════════════════════════════
    static class TrainingStats {
        List<Double> rewards = new ArrayList<>();
        List<Integer> slas   = new ArrayList<>();
        List<Double> cpuLog  = new ArrayList<>();
        List<Double> execLog = new ArrayList<>();
        List<Double> energyLog = new ArrayList<>();
        List<Double> costLog = new ArrayList<>();
        List<Integer> hostLog = new ArrayList<>();
        List<Integer> vmLog = new ArrayList<>();
        int totalSLA = 0;

        void record(double r, int sla, CloudEnv env) {
            rewards.add(r);
            slas.add(sla);
            cpuLog.add(env.avgCPU());
            execLog.add(env.executionTimeMs());
            energyLog.add(env.stepEnergyKWh());
            costLog.add(env.stepCostUsd());
            hostLog.add(env.activeHosts());
            vmLog.add(env.totalVMs());
            totalSLA += sla;
        }

        double avgRewardLast(int n) {
            int start = Math.max(0, rewards.size() - n);
            return rewards.subList(start, rewards.size())
                    .stream().mapToDouble(Double::doubleValue).average().orElse(0);
        }

        double lastReward() { return rewards.isEmpty() ? 0 : rewards.get(rewards.size()-1); }
        int    lastSLA()    { return slas.isEmpty()    ? 0 : slas.get(slas.size()-1); }

        void printFinal(double runtimeMs) {
            double best = rewards.stream().mapToDouble(Double::doubleValue).max().orElse(0);
            double worst = rewards.stream().mapToDouble(Double::doubleValue).min().orElse(0);
            double avg  = rewards.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double avgCpu = cpuLog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double avgExec = execLog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double avgEnergy = energyLog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double totalEnergy = energyLog.stream().mapToDouble(Double::doubleValue).sum();
            double avgCost = costLog.stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double totalCost = costLog.stream().mapToDouble(Double::doubleValue).sum();
            double avgHosts = hostLog.stream().mapToInt(Integer::intValue).average().orElse(0);
            double avgVms = vmLog.stream().mapToInt(Integer::intValue).average().orElse(0);
            System.out.printf("║  Total episodes    : %-5d                              ║%n", rewards.size());
            System.out.printf("║  Avg reward        : %-6.3f                             ║%n", avg);
            System.out.printf("║  Best episode      : %-6.3f                             ║%n", best);
            System.out.printf("║  Worst episode     : %-6.3f                             ║%n", worst);
            System.out.printf("║  Total SLA breaches: %-5d                              ║%n", totalSLA);
            System.out.printf("║  Final avg10 reward: %-6.3f                             ║%n", avgRewardLast(10));
            System.out.printf("║  Avg CPU usage     : %-6.2f%%                            ║%n", avgCpu);
            System.out.printf("║  Avg hosts / VMs   : %-4.1f / %-6.1f                     ║%n", avgHosts, avgVms);
            System.out.printf("║  Avg exec time     : %-6.2f ms                          ║%n", avgExec);
            System.out.printf("║  Avg energy / ep   : %-6.4f kWh                         ║%n", avgEnergy);
            System.out.printf("║  Total energy      : %-6.4f kWh                         ║%n", totalEnergy);
            System.out.printf("║  Avg cost / ep     : $%-7.4f                            ║%n", avgCost);
            System.out.printf("║  Total cost        : $%-7.4f                            ║%n", totalCost);
            System.out.printf("║  Training runtime  : %-8.2f ms                         ║%n", runtimeMs);
        }
    }
}
