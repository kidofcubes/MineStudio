GROOT: Learning to Follow Instructions by Watching Gameplay Videos
======================================================================

.. admonition:: Quick Facts
    
    - GROOT is an open-world controller that follows open-ended instructions by using reference videos as expressive goal specifications, eliminating the need for text annotations. 
    - GROOT leverages a causal transformer-based encoder-decoder architecture to self-supervise the learning of a structured goal space from gameplay videos.

Insights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop an effective instruction-following controller, defining a robust goal representation is essential. Unlike previous approaches, such as using language descriptions or future images (e.g., Steve-1), GROOT leverages reference videos as goal representations. These gameplay videos serve as a rich and expressive source of information, enabling the agent to learn complex behaviors effectively. The paper frames the learning process as future state prediction, allowing the agent to follow demonstrations seamlessly.

Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Formally, the future state prediction problem is defined as maximizing the log-likelihood of future states given past ones: :math:\log p_{\theta}(s_{t+1:T} | s_{0:t}). By introducing :math:`g` as a latent variable conditioned on past states, the evidence lower bound (ELBO) can be expressed as:

.. math::

    \log p_{\theta}(s_{t+1:T} | s_{0:t}) &= \log \sum_g p_{\theta}(s_{t+1:T}, g | s_{0:t}) \\
    &\geq \mathbb{E}_{g \sim q_\phi(\cdot | s_{0:t})} \left[ \log p_{\theta}(s_{t+1:T} | g, s_{0:t}) \right] - D_{\text{KL}}(q_\phi(g | s_{0:T}) || p_\theta(g|s_{0:t})),

where :math:`D_{\text{KL}}` is the Kullback-Leibler divergence, and :math:`q_\phi` represents the variational posterior.

This objective can be further simplified using the transition function :math:`p_{\theta}(s_{t+1}|s_{0:t},a_t)` and a goal-conditioned policy (to be learned) :math:`\pi_{\theta}(a_t|s_{0:t},g)`:

.. math::

    \log p_{\theta}(s_{t+1:T} | s_{0:t}) \geq \sum_{\tau = t}^{T - 1} \mathbb{E}_{g \sim q_\phi(\cdot | s_{0:T}), a_\tau \sim p_{\theta}(\cdot | s_{0:\tau+1})} \left[ \log \pi_{\theta}(a_{\tau} | s_{0:\tau}, g) \right] - D_{\text{KL}}(q_\phi(g | s_{0:T}) || p_\theta(g|s_{0:t})),

where :math:`q_\phi(\cdot|s_{0:T})` is implemented as a video encoder, and :math:`p_{\theta}(\cdot|s_{0:\tau+1})` represents the Inverse Dynamic Model (IDM), which predicts actions to transition to the next state and is typically a pretrained model.
Please refer to the `paper <https://arxiv.org/pdf/2310.08235>`_ for more details.

Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/image/groot_architecture.png
    :width: 800
    :align: center

    GROOT agent architecture.

The GROOT agent consists of a video encoder and a policy decoder.
The video encoder is a non-causal transformer that extracts semantic information and generates goal embeddings.
The policy is a causal transformer
decoder that receives the goal embeddings as the instruction and autoregressively translates the state
sequence into a sequence of actions.
