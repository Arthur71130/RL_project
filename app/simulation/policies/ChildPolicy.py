from app.simulation.policies.Policy import Policy
from app.simulation.envs.Env import Env
import gymnasium as gym

class ChildPolicy(Policy):
    def _predict(self, obs, info):
        action_mask = info.get("action_mask", [])
        if not action_mask:
            return 0

        hold_action = len(action_mask) - 1
        valid_actions = [i for i, is_valid in enumerate(action_mask) if is_valid]

        # No valid service action available, HOLD.
        service_actions = [a for a in valid_actions if a != hold_action]
        if not service_actions:
            return hold_action

        sim_time = info.get("system_time", 0.0)
        env = self.env.unwrapped
        waiting_customers = env.customer_waiting
        appointments = env.appointments

        best_action = hold_action
        best_score = -float("inf")

        for action in service_actions:
            customer_id = int(obs[action])
            if customer_id < 0:
                continue

            customer = waiting_customers.get(customer_id)
            if customer is None:
                continue

            waiting_time = sim_time - customer.arrival_time
            score = waiting_time

            # Strong priority for appointments close to current time.
            appointment = appointments.get(customer_id)
            if appointment is not None:
                appointment_delta = abs(sim_time - appointment.time)
                score += max(0.0, 120.0 - 4.0 * appointment_delta)

                # Penalize taking an appointment too early.
                if sim_time < appointment.time - 30.0:
                    score -= (appointment.time - sim_time - 30.0) * 2.0

            # Small tie-breaker on lower customer id for reproducibility.
            score -= customer.id * 1e-6

            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    
    def learn(self, scenario, total_timesteps, verbose):
        """
        Learning and intialization method for learning models.
        For other models, does nothing.

        Parameters: 
            scenario (scenario): scenario to train
            total_timesteps(int): number of time steps to learn
            verbose(bool): show logs 
        """
        # Heuristic policy: no training phase required.
        _ = gym.make("Child_Env", mode=Env.MODE.TRAIN, scenario=scenario)
        _ = total_timesteps
        if verbose:
            print("ChildPolicy heuristic loaded: no learning step.")
