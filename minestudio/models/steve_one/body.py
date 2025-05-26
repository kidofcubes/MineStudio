from copy import deepcopy
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat

from minestudio.models.base_policy import MinePolicy
from huggingface_hub import PyTorchModelHubMixin

from minestudio.utils.mineclip_lib.mineclip import MineCLIP
from minestudio.utils.vpt_lib.impala_cnn import ImpalaCNN
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.models.base_policy import MinePolicy

class TranslatorVAE(torch.nn.Module):
    """A Variational Autoencoder (VAE) to translate between visual and text embeddings.

    This module can encode pairs of visual and text embeddings into a shared latent space
    and decode from this latent space back to the visual embedding space, conditioned on
    the text embedding.

    :param input_dim: Dimension of the input visual and text embeddings (assumed to be the same).
                      Defaults to 512.
    :type input_dim: int
    :param hidden_dim: Dimension of the hidden layers in the encoder and decoder. Defaults to 256.
    :type hidden_dim: int
    :param latent_dim: Dimension of the latent space. Defaults to 256.
    :type latent_dim: int
    """

    def __init__(self, input_dim=512, hidden_dim=256, latent_dim=256):
        """Initialize the TranslatorVAE.

        :param input_dim: Dimensionality of input embeddings. Defaults to 512.
        :type input_dim: int
        :param hidden_dim: Dimensionality of hidden layers. Defaults to 256.
        :type hidden_dim: int
        :param latent_dim: Dimensionality of the latent space. Defaults to 256.
        :type latent_dim: int
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, visual_embeddings, text_embeddings):
        """Encode the given visual and text embeddings into a latent vector representation (mu and logvar).

        :param visual_embeddings: Tensor of visual embeddings.
        :type visual_embeddings: torch.Tensor
        :param text_embeddings: Tensor of text embeddings.
        :type text_embeddings: torch.Tensor
        :returns: Tensor representing the parameters (mu and logvar concatenated) of the latent distribution.
        :rtype: torch.Tensor
        """
        # Concatenate the visual and text embeddings.
        x = torch.cat([visual_embeddings, text_embeddings], dim=1)
        # Encode the concatenated embeddings into a latent vector.
        return self.encoder(x)

    def sample(self, mu, logvar):
        """Sample a latent vector from the Gaussian distribution defined by mu and logvar.

        Uses the reparameterization trick.

        :param mu: Mean of the latent Gaussian distribution.
        :type mu: torch.Tensor
        :param logvar: Log variance of the latent Gaussian distribution.
        :type logvar: torch.Tensor
        :returns: A sampled latent vector.
        :rtype: torch.Tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_vector, text_embeddings):
        """Decode the given latent vector and text embeddings into a visual embedding.

        :param latent_vector: The latent vector to decode.
        :type latent_vector: torch.Tensor
        :param text_embeddings: Text embeddings to condition the decoding.
        :type text_embeddings: torch.Tensor
        :returns: The reconstructed visual embedding.
        :rtype: torch.Tensor
        """
        # Concatenate the latent vector and text embeddings.
        x = torch.cat([latent_vector, text_embeddings], dim=1)
        # Decode the concatenated embeddings into a visual embedding.
        return self.decoder(x)

    def forward(self, text_embeddings, deterministic=False):
        """Generate a visual embedding from text embeddings using a prior latent distribution.

        This forward pass uses a standard Gaussian prior (mu=0, logvar=0) for the latent space.
        It samples from this prior (or uses the mean if deterministic) and decodes it
        conditioned on the provided text embeddings.

        :param text_embeddings: Text embeddings to condition the generation.
        :type text_embeddings: torch.Tensor
        :param deterministic: If True, use the mean of the prior (mu=0) instead of sampling.
                              Defaults to False.
        :type deterministic: bool
        :returns: The predicted visual embedding.
        :rtype: torch.Tensor
        """
        # Use the prior as the mean and logvar.
        mu = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)
        logvar = torch.zeros(text_embeddings.shape[0], self.latent_dim).to(text_embeddings.device)

        # Sample a latent vector from the mu and logvar.
        if deterministic:
            latent_vector = mu
        else:
            latent_vector = self.sample(mu, logvar)

        # Decode the latent vector into a visual embedding.
        pred_visual_embeddings = self.decode(latent_vector, text_embeddings)

        return pred_visual_embeddings

class ImgPreprocessing(torch.nn.Module):
    """Normalize incoming images.

    :param img_statistics: remote path to npz file with a mean and std image. If specified
        normalize images using this.
    :param scale_img: If true and img_statistics not specified, scale incoming images by 1/255.
    """

    def __init__(self, img_statistics: Optional[str] = None, scale_img: bool = True):
        """Initialize ImgPreprocessing.

        :param img_statistics: Path to a .npz file containing 'mean' and 'std' for image normalization.
                               If None, normalization will be a simple scaling. Defaults to None.
        :type img_statistics: Optional[str]
        :param scale_img: If True and `img_statistics` is None, scale images by 1/255.0.
                          Defaults to True.
        :type scale_img: bool
        """
        super().__init__()
        self.img_mean = None
        if img_statistics is not None:
            img_statistics = dict(**np.load(img_statistics))
            self.img_mean = torch.nn.Parameter(torch.Tensor(img_statistics["mean"]), requires_grad=False)
            self.img_std = torch.nn.Parameter(torch.Tensor(img_statistics["std"]), requires_grad=False)
        else:
            self.ob_scale = 255.0 if scale_img else 1.0

    def forward(self, img):
        """Apply image preprocessing.

        Normalizes the input image tensor. If `img_statistics` was provided during
        initialization, it uses the mean and std from the file. Otherwise, it scales
        the image by `1.0 / self.ob_scale`.

        Note: The input `img` is assumed to be already scaled to [0, 1] if `img_statistics` is used,
        or in [0, 255] if `scale_img` is True and `img_statistics` is None.

        :param img: The input image tensor.
        :type img: torch.Tensor
        :returns: The preprocessed image tensor.
        :rtype: torch.Tensor
        """
        x = img
        if self.img_mean is not None:
            x = (x - self.img_mean) / self.img_std
        else:
            x = x / self.ob_scale
        return x

class ImgObsProcess(torch.nn.Module):
    """ImpalaCNN followed by a linear layer.

    :param cnn_outsize: impala output dimension
    :param output_size: output size of the linear layer.
    :param dense_init_norm_kwargs: kwargs for linear FanInInitReLULayer
    :param init_norm_kwargs: kwargs for 2d and 3d conv FanInInitReLULayer
    """

    def __init__(
        self,
        cnn_outsize: int,
        output_size: int,
        dense_init_norm_kwargs: Dict = {},
        init_norm_kwargs: Dict = {},
        **kwargs,
    ):
        """Initialize ImgObsProcess.

        :param cnn_outsize: The output size of the ImpalaCNN.
        :type cnn_outsize: int
        :param output_size: The final output size after the linear layer.
        :type output_size: int
        :param dense_init_norm_kwargs: Keyword arguments for the dense FanInInitReLULayer (linear layer).
                                       Defaults to {}.
        :type dense_init_norm_kwargs: Dict
        :param init_norm_kwargs: Keyword arguments for the convolutional FanInInitReLULayers (within ImpalaCNN).
                                 Defaults to {}.
        :type init_norm_kwargs: Dict
        :param kwargs: Additional keyword arguments passed to ImpalaCNN.
        """
        super().__init__()
        self.cnn = ImpalaCNN(
            outsize=cnn_outsize,
            init_norm_kwargs=init_norm_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **kwargs,
        )
        self.linear = FanInInitReLULayer(
            cnn_outsize,
            output_size,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )

    def forward(self, img):
        """Process the image observation.

        Passes the image through the ImpalaCNN and then a linear layer.

        :param img: The input image tensor.
        :type img: torch.Tensor
        :returns: The processed image features.
        :rtype: torch.Tensor
        """
        return self.linear(self.cnn(img))

class MinecraftPolicy(torch.nn.Module):
    """
    :param recurrence_type:
        None                - No recurrence, adds no extra layers
        lstm                - (Depreciated). Singular LSTM
        multi_layer_lstm    - Multi-layer LSTM. Uses n_recurrence_layers to determine number of consecututive LSTMs
            Does NOT support ragged batching
        multi_masked_lstm   - Multi-layer LSTM that supports ragged batching via the first vector. This model is slower
            Uses n_recurrence_layers to determine number of consecututive LSTMs
        transformer         - Dense transformer
    :param init_norm_kwargs: kwargs for all FanInInitReLULayers.
    """

    def __init__(
        self,
        recurrence_type="lstm",
        impala_width=1,
        impala_chans=(16, 32, 32),
        obs_processing_width=256, # Unused
        hidsize=512,
        single_output=False,
        img_shape=None,
        scale_input_img=True,
        only_img_input=False, # Unused
        init_norm_kwargs={},
        impala_kwargs={},
        input_shape=None, # Unused
        active_reward_monitors=None, # Unused
        img_statistics=None,
        first_conv_norm=False,
        diff_mlp_embedding=False, # Unused
        attention_mask_style="clipped_causal",
        attention_heads=8,
        attention_memory_size=2048,
        use_pointwise_layer=True,
        pointwise_ratio=4,
        pointwise_use_activation=False,
        n_recurrence_layers=1,
        recurrence_is_residual=True,
        timesteps=None,
        use_pre_lstm_ln=True,
        mineclip_embed_dim=512,
        **unused_kwargs,
    ):
        """Initialize the MinecraftPolicy network for Steve-1.

        This network processes image observations and MineCLIP text embeddings, 
        applies a recurrent layer, and produces latent representations for policy and value functions.

        :param recurrence_type: Type of recurrence. Defaults to "lstm".
        :type recurrence_type: str
        :param impala_width: Width multiplier for ImpalaCNN. Defaults to 1.
        :type impala_width: int
        :param impala_chans: Channels for ImpalaCNN. Defaults to (16, 32, 32).
        :type impala_chans: Tuple[int, ...]
        :param obs_processing_width: (Unused) Width for observation processing. Defaults to 256.
        :type obs_processing_width: int
        :param hidsize: Hidden size for layers. Defaults to 512.
        :type hidsize: int
        :param single_output: If True, policy/value functions share latent. Defaults to False.
        :type single_output: bool
        :param img_shape: Shape of input image. Defaults to None.
        :type img_shape: Optional[Tuple[int, ...]]
        :param scale_input_img: Whether to scale input images. Defaults to True.
        :type scale_input_img: bool
        :param only_img_input: (Unused) Flag for only image input. Defaults to False.
        :type only_img_input: bool
        :param init_norm_kwargs: Kwargs for FanInInitReLULayer norm. Defaults to {}.
        :type init_norm_kwargs: Dict
        :param impala_kwargs: Additional kwargs for ImpalaCNN. Defaults to {}.
        :type impala_kwargs: Dict
        :param input_shape: (Unused) Expected input shape. Defaults to None.
        :type input_shape: Optional[Any]
        :param active_reward_monitors: (Unused) Reward monitor config. Defaults to None.
        :type active_reward_monitors: Optional[Dict]
        :param img_statistics: Path to image normalization statistics. Defaults to None.
        :type img_statistics: Optional[str]
        :param first_conv_norm: Norm after first Impala conv. Defaults to False.
        :type first_conv_norm: bool
        :param diff_mlp_embedding: (Unused) Flag for diff MLP embedding. Defaults to False.
        :type diff_mlp_embedding: bool
        :param attention_mask_style: Attention mask style for Transformer. Defaults to "clipped_causal".
        :type attention_mask_style: str
        :param attention_heads: Num attention heads for Transformer. Defaults to 8.
        :type attention_heads: int
        :param attention_memory_size: Memory size for Transformer attention. Defaults to 2048.
        :type attention_memory_size: int
        :param use_pointwise_layer: Use pointwise layers in recurrent blocks. Defaults to True.
        :type use_pointwise_layer: bool
        :param pointwise_ratio: Ratio for pointwise layer dim. Defaults to 4.
        :type pointwise_ratio: int
        :param pointwise_use_activation: Activation in pointwise layer. Defaults to False.
        :type pointwise_use_activation: bool
        :param n_recurrence_layers: Number of recurrent layers. Defaults to 1.
        :type n_recurrence_layers: int
        :param recurrence_is_residual: Residual connections in recurrent blocks. Defaults to True.
        :type recurrence_is_residual: bool
        :param timesteps: Timesteps for recurrence. Defaults to None.
        :type timesteps: Optional[int]
        :param use_pre_lstm_ln: LayerNorm before recurrent layer. Defaults to True.
        :type use_pre_lstm_ln: bool
        :param mineclip_embed_dim: Dimension of MineCLIP embeddings. Defaults to 512.
        :type mineclip_embed_dim: int
        :param unused_kwargs: Catches other kwargs.
        """
        super().__init__()
        assert recurrence_type in [
            "multi_layer_lstm",
            "multi_layer_bilstm",
            "multi_masked_lstm",
            "transformer",
            "none",
        ]

        active_reward_monitors = active_reward_monitors or {}

        self.single_output = single_output

        # Dense init kwargs replaces batchnorm/groupnorm with layernorm
        self.init_norm_kwargs = init_norm_kwargs
        self.dense_init_norm_kwargs = deepcopy(init_norm_kwargs)
        if self.dense_init_norm_kwargs.get("group_norm_groups", None) is not None:
            self.dense_init_norm_kwargs.pop("group_norm_groups", None)
            self.dense_init_norm_kwargs["layer_norm"] = True
        if self.dense_init_norm_kwargs.get("batch_norm", False):
            self.dense_init_norm_kwargs.pop("batch_norm", False)
            self.dense_init_norm_kwargs["layer_norm"] = True

        # Setup inputs
        self.img_preprocess = ImgPreprocessing(img_statistics=img_statistics, scale_img=scale_input_img)
        self.img_process = ImgObsProcess(
            cnn_outsize=256,
            output_size=hidsize,
            inshape=img_shape,
            chans=tuple(int(impala_width * c) for c in impala_chans),
            nblock=2,
            dense_init_norm_kwargs=self.dense_init_norm_kwargs,
            init_norm_kwargs=init_norm_kwargs,
            first_conv_norm=first_conv_norm,
            **impala_kwargs,
        )

        self.pre_lstm_ln = torch.nn.LayerNorm(hidsize) if use_pre_lstm_ln else None
        self.diff_obs_process = None

        self.recurrence_type = recurrence_type

        self.recurrent_layer = None
        self.recurrent_layer = ResidualRecurrentBlocks(
            hidsize=hidsize,
            timesteps=timesteps,
            recurrence_type=recurrence_type,
            is_residual=recurrence_is_residual,
            use_pointwise_layer=use_pointwise_layer,
            pointwise_ratio=pointwise_ratio,
            pointwise_use_activation=pointwise_use_activation,
            attention_mask_style=attention_mask_style,
            attention_heads=attention_heads,
            attention_memory_size=attention_memory_size,
            n_block=n_recurrence_layers,
        )

        self.lastlayer = FanInInitReLULayer(hidsize, hidsize, layer_type="linear", **self.dense_init_norm_kwargs)
        self.final_ln = torch.nn.LayerNorm(hidsize)

        # MODIFIED (added this)
        self.mineclip_embed_linear = torch.nn.Linear(mineclip_embed_dim, hidsize)

    def output_latent_size(self):
        """Returns the size of the output latent vector.

        :returns: The hidden size, which is the dimension of the output latents.
        :rtype: int
        """
        return self.hidsize

    def forward(self, ob, state_in, context):
        """Forward pass of the MinecraftPolicy (Steve-1 variant).

        Processes image observations ("img") and MineCLIP text embeddings ("mineclip_embed"),
        combines them, passes them through recurrent layers, and produces latent representations.

        :param ob: Dictionary of observations, expected to contain "img" and "mineclip_embed".
        :type ob: Dict[str, torch.Tensor]
        :param state_in: Input recurrent state.
        :type state_in: Any # Type depends on recurrence_type
        :param context: Context dictionary, expected to contain "first" (a tensor indicating episode starts).
        :type context: Dict[str, torch.Tensor]
        :returns: A tuple containing:
            - pi_latent_or_tuple (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                If `single_output` is True, this is a single tensor for both policy and value.
                Otherwise, it's a tuple (pi_latent, vf_latent).
            - state_out (Any): Output recurrent state.
        :rtype: Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Any]
        """
        b, t = ob["img"].shape[:2]
        first = context["first"].bool()

        x = self.img_preprocess(ob["img"])
        x = self.img_process(x)

        if self.diff_obs_process:
            processed_obs = self.diff_obs_process(ob["diff_goal"])
            x = processed_obs + x

        # MODIFIED (added this)
        mineclip_embed = ob["mineclip_embed"].reshape(b * t, -1)
        # Normalize mineclip_embed (doesn't work because the norm is way too small then?)
        # mineclip_embed = F.normalize(mineclip_embed, dim=-1)
        mineclip_embed = self.mineclip_embed_linear(mineclip_embed)
        mineclip_embed = mineclip_embed.reshape(b, t, -1)
        x = x + mineclip_embed

        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in

        x = F.relu(x, inplace=False)

        x = self.lastlayer(x)
        x = self.final_ln(x)
        pi_latent = vf_latent = x
        if self.single_output:
            return pi_latent, state_out
        return (pi_latent, vf_latent), state_out

    def initial_state(self, batchsize):
        """Get the initial recurrent state.

        :param batchsize: The batch size for the initial state.
        :type batchsize: int
        :returns: The initial recurrent state, or None if no recurrent layer is used.
        :rtype: Any # Type depends on recurrence_type, can be None
        """
        if self.recurrent_layer:
            return self.recurrent_layer.initial_state(batchsize)
        else:
            return None

class SteveOnePolicy(MinePolicy, PyTorchModelHubMixin):
    """Steve-1 policy model, combining visual processing with language conditioning via MineCLIP.

    This policy uses a `MinecraftPolicy` core (which includes an ImpalaCNN and recurrent layers)
    and integrates MineCLIP for processing text prompts. It can optionally use a `TranslatorVAE`
    to translate between text and visual embedding spaces for more nuanced conditioning.

    :param policy_kwargs: Keyword arguments for the underlying `MinecraftPolicy`.
    :type policy_kwargs: Dict
    :param mineclip_kwargs: Keyword arguments for initializing MineCLIP.
    :type mineclip_kwargs: Dict
    :param trans_kwargs: Keyword arguments for initializing `TranslatorVAE`. If None, VAE is not used.
                         Defaults to None.
    :type trans_kwargs: Optional[Dict]
    :param action_space: The action space definition. Passed to `MinePolicy`.
    :type action_space: Optional[Any]
    :param internal_hiddim: Hidden dimension for policy/value heads. Passed to `MinePolicy` as `hiddim`.
    :type internal_hiddim: int
    :param kwargs: Additional keyword arguments for `MinePolicy` (e.g., temperature).
    """
    
    def __init__(
        self,
        policy_kwargs: Dict[str, Any],
        mineclip_kwargs: Dict[str, Any],
        trans_kwargs: Optional[Dict[str, Any]] = None,
        action_space=None,
        internal_hiddim=None,
        **kwargs
    ):
        """Initialize the SteveOnePolicy.

        :param policy_kwargs: Keyword arguments for the `MinecraftPolicy` core.
        :type policy_kwargs: Dict[str, Any]
        :param mineclip_kwargs: Keyword arguments for MineCLIP.
        :type mineclip_kwargs: Dict[str, Any]
        :param trans_kwargs: Keyword arguments for `TranslatorVAE`. If None, VAE is disabled.
                             Defaults to None.
        :type trans_kwargs: Optional[Dict[str, Any]]
        :param action_space: Action space definition.
        :type action_space: Optional[Any]
        :param internal_hiddim: Hidden dimension for policy/value heads. If None, uses `policy_kwargs['hidsize']`.
        :type internal_hiddim: Optional[int]
        :param kwargs: Additional keyword arguments for `MinePolicy`.
        """
        if internal_hiddim is None:
            internal_hiddim = policy_kwargs["hidsize"]
        super().__init__(hiddim=internal_hiddim, action_space=action_space, **kwargs)

        self.net = MinecraftPolicy(**policy_kwargs)
        self.mineclip = MineCLIP(**mineclip_kwargs)
        self.translator_vae = TranslatorVAE(**trans_kwargs) if trans_kwargs is not None else None

        # Cache for MineCLIP text embeddings
        self._mineclip_cache = {}

    def _get_mineclip_attns(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Computes MineCLIP text embeddings for a given prompt, using a cache.

        If the prompt has been processed before, returns the cached embedding.
        Otherwise, computes it, caches it, and returns it.

        :param prompt: The text prompt.
        :type prompt: str
        :param device: The torch device to move the embedding to.
        :type device: torch.device
        :returns: The MineCLIP text embedding for the prompt.
        :rtype: torch.Tensor
        """
        # Create a hash of the prompt to use as a cache key.
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if prompt_hash in self._mineclip_cache:
            attn = self._mineclip_cache[prompt_hash]
        else:
            # If the prompt is new, compute the embedding.
            if isinstance(prompt, list):
                attn = [self.mineclip.encode_text(p).detach() for p in prompt]
            else:
                attn = self.mineclip.encode_text(prompt).detach()

            # Cache the computed embedding.
            self._mineclip_cache[prompt_hash] = attn

        if isinstance(attn, list):
            attn = torch.stack(attn)

        return attn.to(device)

    def forward(
        self, 
        input: Dict[str, Any], 
        state_in: Optional[List[torch.Tensor]] = None, 
        **kwargs
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """Forward pass of the SteveOnePolicy.

        Processes image observations and a text prompt. The text prompt is converted to a
        MineCLIP embedding. If a TranslatorVAE is used, this embedding can be further
        transformed. The visual features and text-derived features are then passed to the
        core `MinecraftPolicy` network.

        :param input: Dictionary of input observations. Expected to contain:
            - "img" (torch.Tensor): Image observations.
            - "prompt" (str or List[str]): Text prompt(s).
        :type input: Dict[str, Any]
        :param state_in: Input recurrent state. If None, an initial state is generated.
        :type state_in: Optional[List[torch.Tensor]]
        :param kwargs: Additional keyword arguments.
        :returns: A tuple containing:
            - latents (Dict[str, torch.Tensor]): Dictionary with 'pi_logits' and 'vpred'.
            - state_out (List[torch.Tensor]): Output recurrent state.
        :rtype: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]
        """
        # MODIFIED (pass mineclip embeddings to the policy)
        if isinstance(input["prompt"], str):
            input["prompt"] = [input["prompt"]]
        mineclip_embeds = self._get_mineclip_attns(input["prompt"][0], device=self.device)

        if self.translator_vae is not None:
            # If using TranslatorVAE, decode the text embeddings to get visual embeddings
            with torch.no_grad():
                visual_embeds = self.translator_vae.decode(mineclip_embeds, mineclip_embeds)
        else:
            visual_embeds = mineclip_embeds

        if state_in is None:
            state_in = self.initial_state(len(input["img"]))

        return self.net(
            ob={
                "img": input["img"],
                "mineclip_embed": visual_embeds,
            },
            state_in=state_in,
            **kwargs
        )

    def initial_state(self, batch_size: Optional[int] = None):
        """Get the initial recurrent state for a given batch size.

        Caches initial states for frequently used batch sizes.

        :param batch_size: The batch size. If None, returns state for batch size 1 (squeezed).
                           Defaults to None.
        :type batch_size: Optional[int]
        :returns: A list of initial state tensors for the recurrent network, moved to the correct device.
        :rtype: List[torch.Tensor]
        """
        if batch_size is None:
            batch_size = 1

        if batch_size in self._initial_state_cache:
            return self._initial_state_cache[batch_size]

        state = self.net.initial_state(batch_size)
        # Move to correct device
        state = [s.to(self.device) for s in state]

        self._initial_state_cache[batch_size] = state
        return state

    def reset_parameters(self):
        super().reset_parameters()
        self.net.reset_parameters()
        raise NotImplementedError()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

def load_steve_one_policy(ckpt_path: str) -> SteveOnePolicy:
    return SteveOnePolicy.from_pretrained(ckpt_path)

if __name__ == '__main__':
    model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to("cuda")
    model.eval()
    condition = model.prepare_condition(
        {
            'cond_scale': 4.0,
            'video': np.random.randint(0, 255, (2, 16, 224, 224, 3)).astype(np.uint8)
        }
    )
    output, memory = model(condition,
        input={
            'image': torch.zeros(2, 8, 128, 128, 3).to("cuda"), 
        },
        state_in=model.initial_state(condition, 2)
    )