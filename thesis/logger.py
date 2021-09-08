class Logger:
    def __init__(self):
        self.log_reward = []
        self.log_q_taken = []
        self.log_critic_loss = []
        self.log_actor_loss = []
        self.log_actor_loss_amount = []
        self.log_entropy = []
        self.log_residual_variance = []

    def append(self, reward, q_taken, critic_loss, actor_loss, actor_loss_amount, entropy, residual_variance):
        self.log_reward.append(reward)
        self.log_q_taken.append(q_taken)
        self.log_critic_loss.append(critic_loss)
        self.log_actor_loss.append(actor_loss)
        self.log_actor_loss_amount.append(actor_loss_amount)
        self.log_entropy.append(entropy)
        self.log_residual_variance.append(residual_variance)

    def get_log(self):
        return (self.log_reward, self.log_q_taken, self.log_critic_loss, self.log_actor_loss,
                self.log_actor_loss_amount, self.log_entropy, self.log_residual_variance)
