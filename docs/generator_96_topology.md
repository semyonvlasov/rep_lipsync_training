# Generator 96x96 Topology

This is a living note for the current `96x96` generator used in `rep_lipsync_training`.
We will extend this file incrementally as new questions come up.

## Scope

This document describes only the current `img_size == 96` generator path in:

- `training/models/generator.py`

It does not describe the generalized `128/192/256` path yet.

## Current Runtime Context

The currently used training config is:

- `training/configs/generator_official_syncnet_continue_from_ablation_winner_20260328.yaml`

Relevant settings:

- `img_size: 96`
- `base_channels: 32`
- `predict_alpha: false`

Important nuance:

- for `img_size == 96`, the generator uses a dedicated hardcoded branch
- this branch closely mirrors the official `Wav2Lip` `96x96` architecture
- `base_channels` is effectively not the driving width parameter here, because the `96` path uses fixed channel widths

Reference official implementation:

- `../models/wav2lip/models/wav2lip.py`

## Input Tensors

At training time the generator receives two inputs.

### Visual Input

`face_input`

- shape: `(B, 6, T, 96, 96)`
- usually `T = 5`

Channel meaning:

- first `3` channels: masked target window
- next `3` channels: reference window

So for one sample the visual input is:

- `5` target frames with the lower half masked out
- `5` reference frames from the same identity

### Audio Input

`indiv_mels`

- shape: `(B, T, 1, 80, 16)`

Meaning:

- `T = 5` mel chunks
- each chunk is one `80 x 16` mel patch
- one chunk is aligned to one frame position in the `T=5` window

## Time Flattening Before Convs

The generator itself is not a `Conv3d` model.
It flattens time into the batch dimension before entering the face/audio towers.

### Visual

```text
(B, 6, T, 96, 96) -> (B*T, 6, 96, 96)
```

### Audio

```text
(B, T, 1, 80, 16) -> (B*T, 1, 80, 16)
```

So if:

- `B = 32`
- `T = 5`

then the actual forward pass processes:

- `160` visual tensors of shape `(6, 96, 96)`
- `160` audio tensors of shape `(1, 80, 16)`

## High-Level Flow

```text
face_input
  -> face encoder
  -> encoder skip features

indiv_mels
  -> audio encoder
  -> audio embedding (512, 1, 1)

audio embedding
  -> face decoder
  -> after each decoder block: concat with matching encoder skip

decoder output
  -> output head
  -> RGB face

reshape back
  -> (B, 3, T, 96, 96)
```

## Building Blocks

### ConvBlock

Used throughout encoder, audio tower, decoder head.

Structure:

```text
Conv2d -> BatchNorm2d -> ReLU
```

Optional residual mode:

- if `residual=True`, output becomes `ReLU(conv(x) + x)`

### How To Read `ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)`

Example:

```text
ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)
```

Meaning:

- `6 -> 16`
  - input tensor has `6` channels
  - output tensor has `16` channels
  - in other words, the convolution maps each spatial location from a `6`-dimensional feature vector to a `16`-dimensional feature vector
  - for the raw face input, those `6` channels are:
    - `3` channels from the masked target frame
    - `3` channels from the reference frame
  - the `16` here is a design choice: it is the width of the first feature layer
  - it means the network is allowed to build `16` different learned feature maps from the same `6` input channels
  - each of those `16` output channels can specialize in a different low-level pattern:
    - edges
    - color transitions
    - mouth contours
    - identity/texture cues from the reference half of the input
  - there is no single mathematically forced reason it must be `16`
  - in practice this number is chosen as a tradeoff:
    - larger than `6`, so the network can expand the representation
    - still small enough to keep the very first high-resolution layer cheap
  - in the official `96x96` Wav2Lip topology, this first width is also `16`, and we keep it for compatibility

- `kernel=7`
  - each output value looks at a `7 x 7` spatial neighborhood
  - because there are `6` input channels, one filter actually has shape:

```text
6 x 7 x 7
```

  - because we produce `16` output channels, the full conv weight tensor has shape:

```text
16 x 6 x 7 x 7
```

This is the shape of the **weight tensor**, not the output activation tensor.

Important distinction:

- input activation tensor:

```text
(B*T, 6, 96, 96)
```

- convolution weights:

```text
(16, 6, 7, 7)
```

- output activation tensor:

```text
(B*T, 16, 96, 96)
```

So this layer does **not** turn:

```text
(B*T, 6, 96, 96) -> (16, 6, 7, 7)
```

Instead it uses weights of shape:

```text
(16, 6, 7, 7)
```

to compute an output activation of shape:

```text
(B*T, 16, 96, 96)
```

Another way to read it:

- there are `16` learned filters
- each filter sees all `6` input channels
- each filter has spatial size `7 x 7`
- each filter produces one output channel
- therefore the output has `16` channels

How one such filter is applied:

- take one output filter of shape `6 x 7 x 7`
- split it conceptually into `6` channel-specific slices:

```text
1 x 7 x 7
1 x 7 x 7
1 x 7 x 7
1 x 7 x 7
1 x 7 x 7
1 x 7 x 7
```

- each `7 x 7` slice is applied to its matching input channel
- this produces `6` partial responses at the same spatial location
- those `6` partial responses are then summed together
- then bias is added
- then the result becomes one scalar in one output channel

So the channel slices are not independent in the final output value.

More precisely:

- they are applied separately to different input channels
- but their results are immediately combined by summation
- therefore one output activation depends jointly on all `6` input channels

For one output filter `o`, the value at spatial position `(i, j)` is:

```text
y[o, i, j] =
  b[o] +
  sum over input channels c and kernel offsets u, v of
  W[o, c, u, v] * x[c, i+u, j+v]
```

So if the question is:

- "does each `7 x 7` channel slice have its own weights?" -> yes
- "does each slice produce a completely separate final output?" -> no
- "are the 6 channel slices merged into one output channel by summation?" -> yes

What kind of summation is this?

- just ordinary scalar addition
- there is no extra learned gate at this step
- there is no averaging
- there is no max operation

At one spatial location, one output channel is computed like this:

1. take the `7 x 7` patch from input channel `0`
2. multiply it elementwise by that channel's `7 x 7` kernel slice
3. sum all `49` products -> this gives one scalar
4. repeat the same for input channels `1..5`
5. add the resulting `6` scalars together
6. add the bias term

What is `bias` here?

- just one learned scalar per output channel
- for this first conv layer there are `16` such bias values total
- one bias belongs to one output filter
- that same bias is added to every spatial position of that output channel

So for the first layer:

- weights shape:

```text
(16, 6, 7, 7)
```

- bias shape:

```text
(16,)
```

Where does it come from?

- it starts as a trainable parameter of the layer
- during training gradient descent updates it just like the conv weights
- so it is not computed from the image directly
- it is learned

So the structure is:

```text
output_value =
  dot(channel0_patch, channel0_kernel)
+ dot(channel1_patch, channel1_kernel)
+ dot(channel2_patch, channel2_kernel)
+ dot(channel3_patch, channel3_kernel)
+ dot(channel4_patch, channel4_kernel)
+ dot(channel5_patch, channel5_kernel)
+ bias
```

Tiny toy example:

- suppose each channel-specific `7 x 7` slice has already produced these partial scalars:

```text
2.1, -0.4, 0.7, 1.3, -0.2, 0.5
```

- and bias is:

```text
0.1
```

Then the pre-activation output is just:

```text
2.1 - 0.4 + 0.7 + 1.3 - 0.2 + 0.5 + 0.1 = 4.1
```

This value is the raw convolution output, often called the pre-activation.

After that, the next operations in our `ConvBlock` are:

- `BatchNorm2d`
- `ReLU`

So the precise order for a non-residual `ConvBlock` is:

```text
Conv2d
-> BatchNorm2d
-> ReLU
```

For the first layer `ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)` this means:

1. `Conv2d` computes a raw activation tensor:

```text
(B*T, 16, 96, 96)
```

2. `BatchNorm2d(16)` normalizes each of the `16` output channels using batch statistics
3. `ReLU` clamps negative values to `0`

### What `BatchNorm2d` Does Here

For one output channel `k`, BatchNorm during training computes over the whole mini-batch and spatial grid:

- a batch mean `mu[k]`
- a batch standard deviation `sigma[k]`

Then for every value in that output channel:

```text
z_hat = (z - mu[k]) / sqrt(sigma[k]^2 + eps)
z_out = gamma[k] * z_hat + beta[k]
```

Very short reading of the formula:

- `z` = raw number after convolution
- `mu` = average value of that output channel over the batch
- `sigma` = standard deviation of that output channel over the batch
- `eps` = tiny constant so we do not divide by zero
- `gamma` = learned scale
- `beta` = learned shift

So in plain language:

1. subtract the channel average
2. divide by the channel spread
3. stretch/compress with `gamma`
4. shift with `beta`

Important clarification:

- this does **not** force values into `(-1, 1)`
- after subtracting mean and dividing by standard deviation, the channel becomes roughly:
  - mean near `0`
  - standard deviation near `1`
- but values can still easily be smaller than `-1` or larger than `1`

What exactly are `mu` and `var` in `BatchNorm2d`?

- for one output channel, `mu` is the average over:
  - all items in the mini-batch
  - all spatial positions in that channel map
- so for a tensor of shape:

```text
(B*T, 16, 96, 96)
```

- the mean for channel `k` is computed over all values in:

```text
[:, k, :, :]
```

So yes:

- it is computed using the whole current batch for that channel
- but not to preserve "filter saturation" directly

The main purpose is:

- keep activation scale more stable from batch to batch
- avoid some channels becoming numerically much larger or smaller than others
- make optimization easier for deeper networks

What BatchNorm preserves vs changes:

- it preserves the ordering inside a channel
  - larger values stay larger than smaller values after normalization
- it changes the absolute scale and offset of the channel
- then `gamma` and `beta` let the network learn how much scale/shift it actually wants back

### Does The Same Pattern Change Value Across Different Batches?

Yes, during training this can happen.

If a filter responds to something like:

- a straight edge
- a mouth contour
- a bright-to-dark transition

then the normalized value for that same local pattern can shift a bit depending on:

- how strong this channel is on average in the current batch
- how many similar responses appear elsewhere in the current batch
- the current batch variance for that channel

So your intuition is basically correct:

- the same "line" can end up with a somewhat different normalized value in different batches

But a few important caveats:

- the effect is channel-wise, not pattern-wise
  - BatchNorm does not count "how many lines" explicitly
  - it only sees the distribution of numbers in that output channel
- it is usually not a wild random jump
  - it is a rescaling relative to the batch mean and spread
- in our setup the statistics are computed over a lot of values:
  - batch axis `B*T`
  - and all `96 x 96` spatial positions
  - so the estimates are usually less noisy than they would be with a tiny tensor

At inference time this changes:

- BatchNorm does **not** use the current batch statistics
- it uses stored running averages collected during training
- so for inference the same input gives a stable deterministic output

So the short practical summary is:

- during training: yes, BatchNorm introduces some batch dependence
- during inference: no, it uses frozen running stats instead

Meaning:

- subtract channel mean
- divide by channel standard deviation
- then apply a learned scale `gamma`
- then apply a learned shift `beta`

Important:

- BatchNorm is done independently for each output channel
- it does not mix the `16` channels together
- it rescales each channel separately

Why this helps:

- stabilizes activation scale
- makes training less sensitive to raw conv magnitude
- helps deeper stacks train more reliably

### What `ReLU` Does Here

After BatchNorm, ReLU is applied elementwise:

```text
ReLU(a) = max(0, a)
```

So:

- positive values stay
- negative values become `0`

This introduces nonlinearity.
Without it, a stack of convolutions would collapse into an effectively linear mapping.

### Residual vs Non-Residual ConvBlock

The very first layer is **not** residual.

So for `E0` the exact computation is:

```text
x
-> Conv2d
-> BatchNorm2d
-> ReLU
```

For a residual `ConvBlock(..., residual=True)` later in the network, the code path is:

```text
y = Conv2d(x)
y = BatchNorm2d(y)
out = ReLU(y + x)
```

So residual blocks differ in one important way:

- they add the input tensor back before the final ReLU

Important clarification:

- these residual connections are **not** the same thing as the encoder-decoder skip connections
- they solve a different problem

There are two separate mechanisms in this generator:

1. residual connection inside a block:

```text
out = ReLU(ConvBN(x) + x)
```

- this happens locally inside one stage
- input and output have the same shape
- purpose:
  - make optimization easier
  - let the block learn a correction instead of a full rewrite
  - preserve information more easily through depth

2. encoder-decoder skip connection:

- save an encoder feature map such as:

```text
32 x 48 x 48
```

- later concatenate it into the decoder at the matching scale
- purpose:
  - bring back spatial information
  - help the decoder know where structures are
  - preserve fine and medium layout details

So the short answer is:

- our residual blocks are **not needed for** the U-Net-style skip connections
- they are a separate local trick for stable deep feature refinement
- the long-range "where is the mouth/face structure" skips come from the stored encoder feature maps, not from `residual=True`

### Is This The Same General Idea As ResNet?

Yes, this is the same general residual-learning idea as in ResNet.

But it is slightly more precise to say:

- the block does not try to learn the full output mapping from scratch
- instead it learns a correction on top of its input

So the mental model is:

```text
out = x + correction(x)
```

not:

```text
out = something completely new, unrelated to x
```

In our simple residual `ConvBlock`, the correction is produced by:

```text
correction(x) = Conv2d(x) -> BatchNorm2d -> then added back to x
```

and only after that:

```text
ReLU(correction(x) + x)
```

So your intuition is right in spirit:

- residual learning means the block starts from the existing representation `x`
- and learns how to modify it

The part I would phrase differently is:

- not "starts from the first conv layer in the pack"
- but "starts from the block input `x` and learns an additive update to it"

- `stride=1`
  - the `7 x 7` filter window moves by `1` pixel at a time
  - this means there is no spatial downsampling in this layer
  - if the input is `96 x 96`, the output can stay `96 x 96` as long as padding is chosen correctly

- `padding=3`
  - add `3` pixels of zero-padding on each side
  - for `kernel=7`, this is exactly what preserves spatial size when `stride=1`
  - important: this does **not** mean kernel windows stop overlapping
  - with `stride=1`, neighboring `7 x 7` windows overlap very heavily
  - padding only extends the image with zeros near the border so that edge pixels can also be seen by a full `7 x 7` neighborhood

What padding is **not** doing here:

- it does not separate kernel windows from each other
- it does not prevent overlap between neighboring receptive fields
- it does not mean the model "cuts the image into non-intersecting squares"

What padding **is** doing here:

- it lets the filter slide over border locations without shrinking the output
- it makes the first layer treat edge regions more symmetrically with center regions

Small intuition:

- without padding, a `7 x 7` convolution on `96 x 96` with `stride=1` would produce `90 x 90`
- with `padding=3`, the effective input becomes `102 x 102`, so the same sliding window gives back `96 x 96`

Overlap intuition:

- the first output position and the next output position are only `1` pixel apart because `stride=1`
- so their `7 x 7` windows overlap in `6` columns
- this heavy overlap is normal and desirable in CNNs

Spatial size formula:

```text
out = floor((in + 2*padding - kernel) / stride) + 1
```

For this layer:

```text
out = floor((96 + 2*3 - 7) / 1) + 1
    = floor(95) + 1
    = 96
```

So:

```text
(B*T, 6, 96, 96) -> (B*T, 16, 96, 96)
```

### Why The First Layer Uses `kernel=7`

Short answer:

- this is an architecture choice, not a hard rule of convolutional networks
- in our `96x96` path we keep `7x7` because the official `Wav2Lip` `96x96` model also starts with `7x7`
- there is no universal theorem saying the first kernel must be `7`

What is generally common in CNNs:

- `3x3` is the most common default kernel size
- `5x5` or `7x7` are often used in the very first layer or stem
- later layers are usually smaller, because stacking several `3x3` layers is often more parameter-efficient than using large kernels everywhere

Why a larger first kernel can make sense:

- the first layer sees raw RGB-style input, not higher-level features yet
- a larger kernel lets the model combine a wider local neighborhood immediately
- for faces, this can help capture:
  - coarse edges
  - mouth contours
  - cheek/jaw boundaries
  - broad color and illumination transitions

Why not use `7x7` everywhere:

- larger kernels cost more parameters and FLOPs
- after the first few layers, the receptive field already grows through depth
- so repeated `3x3` layers are usually a better tradeoff

So was `7x7` experimentally tuned?

- most likely it was chosen as a reasonable stem design, not as a mathematically special value
- we do not have evidence in the official code or paper that `7x7` was the result of a dedicated kernel-size ablation
- the safest statement is:
  - `7x7` is a conventional and sensible first-layer choice
  - and we keep it mainly because it matches the official `96x96` architecture

### DeconvBlock

Used in decoder upsampling blocks.

Structure:

```text
ConvTranspose2d -> BatchNorm2d -> ReLU
```

## Face Encoder

The face encoder compresses:

```text
(B*T, 6, 96, 96) -> (B*T, 512, 1, 1)
```

It also stores skip tensors at every resolution stage.

### E0

```text
ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)
```

Output:

```text
(B*T, 16, 96, 96)
```

### E1

```text
ConvBlock(16 -> 32, kernel=3, stride=2, padding=1)
ConvBlock(32 -> 32, residual=True)
ConvBlock(32 -> 32, residual=True)
```

Output:

```text
(B*T, 32, 48, 48)
```

What this sequence is doing:

1. first block:
   - changes channels `16 -> 32`
   - downsamples spatial size `96 -> 48`
   - this is the moment where the network moves from very raw low-level features to a slightly richer and cheaper representation

2. second block:
   - keeps shape `32 x 48 x 48`
   - refines features at the same resolution
   - because it is residual, it learns a correction on top of the current representation, not a full replacement

3. third block:
   - again keeps shape `32 x 48 x 48`
   - does another refinement step at the same scale
   - again as a residual update

So the role of `E1` is:

- do the first real downsampling
- expand the feature width
- then spend a little extra compute refining the representation before the next downsampling stage

Very short intuition:

- first block = "shrink and widen"
- next two residual blocks = "polish at the same scale"

Why use `stride=2` in the first block?

- this halves height and width:

```text
96 x 96 -> 48 x 48
```

- after that, later convolutions are much cheaper
- and each feature can represent a slightly larger area of the original face

Why are the next two blocks residual?

- a residual block computes:

```text
out = ReLU(ConvBN(x) + x)
```

- instead of forcing the layer to build a totally new representation from scratch, it can learn:
  - "keep most of x"
  - "just add this useful correction"

Why this is helpful:

- optimization is easier
- deeper stacks train more stably
- feature refinement at the same resolution becomes cheaper to learn conceptually

Plain-language intuition for residual:

- non-residual block says: "replace this feature map with a new one"
- residual block says: "keep the old feature map, but tweak it a bit"

For `E1`, this means:

- after the first downsample to `32 x 48 x 48`
- the next two blocks mostly refine edges, contours, and local face patterns
- without changing the tensor shape

Why increase channels when spatial size is reduced?

Short answer:

- this is a very common CNN design heuristic
- but it is **not** a mandatory law

The intuition is:

- after downsampling, the map has fewer spatial positions
- so each remaining position is allowed to carry a richer feature vector
- in other words:
  - fewer locations
  - more information per location

Here the transition is:

```text
16 x 96 x 96
-> 32 x 48 x 48
```

What changes:

- width and height each shrink by `2`
- total number of spatial positions shrinks by `4`
- channels grow only by `2`

So the representation becomes:

- spatially coarser
- channel-wise richer

This often makes sense because deeper layers care less about exact pixel placement and more about higher-level combinations of features.

Is `x2 channels when /2 spatial` required?

- no
- it is just a common rule of thumb

Other valid choices would be:

- keep channels the same
- increase them by less than `2x`
- increase them by more than `2x`

Those choices trade off:

- compute
- memory
- representation capacity
- training stability

Why `x2` is a popular compromise:

- if spatial area drops by `4x`, we can afford somewhat wider channels
- doubling channels partly compensates for lost spatial detail
- it often keeps later compute in a reasonable range

Very rough intuition:

- high resolution layers are expensive because `H x W` is large
- once `H` and `W` are cut in half, widening channels becomes much cheaper than it was one stage earlier

So the design idea is not:

- "we must double channels"

It is more like:

- "after shrinking the map, we can spend some of the saved compute on richer features"

### E2

```text
ConvBlock(32 -> 64, kernel=3, stride=2, padding=1)
ConvBlock(64 -> 64, residual=True)
ConvBlock(64 -> 64, residual=True)
ConvBlock(64 -> 64, residual=True)
```

Output:

```text
(B*T, 64, 24, 24)
```

What `E2` is doing:

1. first block:
   - downsamples `48 x 48 -> 24 x 24`
   - widens channels `32 -> 64`
   - moves the representation one level deeper and more abstract

2. next three residual blocks:
   - keep the same shape `64 x 24 x 24`
   - spend extra compute refining features at this scale
   - do not change resolution

Short intuition:

- first block = "shrink again and widen again"
- next three residual blocks = "do more thinking at this scale"

Why are there `3` residual blocks here instead of `2`?

There is no hard mathematical rule saying this stage must have exactly `3`.
This is an architecture choice.

The likely intuition is:

- by `24 x 24`, the feature maps are already much cheaper than at `48 x 48`
- so the network can afford more same-resolution processing here
- this stage is also deep enough that the features are starting to represent more useful combinations of local face structure

Compared with `E1`:

- `E1` is still quite close to raw image features
- `E2` is a more comfortable point to spend extra capacity on refinement

So `E2` gets one extra residual block because:

- compute is cheaper than at higher resolutions
- feature abstraction is richer than in `E1`
- the architecture designer chose to allocate more processing depth here

What those residual blocks are likely refining at this stage:

- contours of mouth and jaw
- local shape groupings rather than raw edges
- combinations of target/reference facial cues

Practical summary:

- the first block in `E2` changes the scale
- the next three blocks improve feature quality at the new scale
- the reason there are `3` of them is architectural capacity allocation, not a mandatory formula

### E3

```text
ConvBlock(64 -> 128, kernel=3, stride=2, padding=1)
ConvBlock(128 -> 128, residual=True)
ConvBlock(128 -> 128, residual=True)
```

Why does `E3` have only `2` residual blocks even though the map is smaller?

Because "smaller map" is only half of the compute story.

At `E3`:

- spatial size is smaller:

```text
24 x 24 -> 12 x 12
```

- but channels are larger:

```text
64 -> 128
```

For a same-shape residual block, rough convolution cost scales like:

```text
H x W x C x C x kernel^2
```

So compare one residual block in `E2` vs one residual block in `E3`:

- `E2` residual block:

```text
24 x 24 x 64 x 64 x 3 x 3
```

- `E3` residual block:

```text
12 x 12 x 128 x 128 x 3 x 3
```

If you simplify:

- `24 x 24 x 64 x 64 = 576 x 4096`
- `12 x 12 x 128 x 128 = 144 x 16384`

These are the same product.

So one `128@12x12` residual block is roughly as expensive as one `64@24x24` residual block.

That means:

- `E3` is not automatically "cheap just because the map is smaller"
- doubling channels roughly cancels out the spatial saving

So why only `2` blocks here?

- again, this is an architecture choice, not a rule
- the likely idea is that the model has already spent a lot of refinement capacity in `E2`
- once features become more compressed and abstract, the designer chose not to keep increasing same-scale depth further

Short practical intuition:

- `E2`: give the `24x24` stage a little extra depth
- `E3`: go deeper in abstraction, but do not keep stacking refinement forever

So the key answer is:

- `E3` has fewer residual blocks not because it is "too small to need them"
- but because each residual block there is still roughly as expensive as in `E2`, and the architecture decided that `2` was enough at that scale

Output:

```text
(B*T, 128, 12, 12)
```

### E4

```text
ConvBlock(128 -> 256, kernel=3, stride=2, padding=1)
ConvBlock(256 -> 256, residual=True)
ConvBlock(256 -> 256, residual=True)
```

Output:

```text
(B*T, 256, 6, 6)
```

### E5

```text
ConvBlock(256 -> 512, kernel=3, stride=2, padding=1)
ConvBlock(512 -> 512, residual=True)
```

Output:

```text
(B*T, 512, 3, 3)
```

### E6

```text
ConvBlock(512 -> 512, kernel=3, stride=1, padding=0)
ConvBlock(512 -> 512, kernel=1, stride=1, padding=0)
```

Output:

```text
(B*T, 512, 1, 1)
```

### Encoder Skip Stack

The stored visual skip tensors are:

```text
E0: 16 x 96 x 96
E1: 32 x 48 x 48
E2: 64 x 24 x 24
E3: 128 x 12 x 12
E4: 256 x 6 x 6
E5: 512 x 3 x 3
E6: 512 x 1 x 1
```

### Why Collapse To `1x1` Instead Of Flattening Earlier?

This is a very good question.

If we looked only at the final encoder output:

```text
512 x 1 x 1
```

then your objection would be correct:

- this tensor no longer contains explicit spatial coordinates
- it is a global summary
- by itself it mostly says "what is present" rather than "exactly where it is"

So why does the architecture still do this?

Because the model does **not** rely on the `1x1` bottleneck alone.

It also keeps the whole skip stack:

```text
16 x 96 x 96
32 x 48 x 48
64 x 24 x 24
128 x 12 x 12
256 x 6 x 6
512 x 3 x 3
512 x 1 x 1
```

Those skip tensors are exactly where most of the spatial information survives.

So the practical split of responsibilities is:

- `1x1` bottleneck:
  - global summary
  - coarse identity/pose/context information
  - compact fusion point with the audio embedding, which is also `512 x 1 x 1`

- skip tensors:
  - where contours are
  - where mouth boundaries are
  - where face structure sits in the frame
  - fine and medium spatial layout

So the decoder does **not** reconstruct the face from a locationless vector alone.
It reconstructs from:

- a global bottleneck
- plus spatial skip features re-injected at every scale

Why not flatten at an earlier stage instead?

Example:

- at `E2`, flattening would mean:

```text
64 x 24 x 24 = 36864
```

That creates a large dense vector per frame.

Problems with that choice:

- it is much heavier than continuing with convolutions
- it breaks the clean 2D grid structure early
- dense layers after flatten are usually much more parameter-hungry
- translation/locality structure is weakened
- it becomes harder to decode back into an image cleanly

Why the cascade of downsampling convs is attractive:

- it gradually increases receptive field
- it preserves local neighborhood structure for a long time
- it compresses information progressively instead of collapsing the whole map at once
- it lets the decoder reuse multi-scale spatial skips

So the honest short answer is:

- yes, the final `512 x 1 x 1` by itself has almost no explicit "where" information
- but the network gets away with this because "where" is carried mostly by the skip connections
- the `1x1` bottleneck is mainly a compact global code, not the sole source of reconstruction information

## Audio Encoder

The audio encoder compresses:

```text
(B*T, 1, 80, 16) -> (B*T, 512, 1, 1)
```

### A0

```text
ConvBlock(1 -> 32, kernel=3, stride=1, padding=1)
```

Output:

```text
(B*T, 32, 80, 16)
```

### A1

```text
ConvBlock(32 -> 32, residual=True)
```

Output:

```text
(B*T, 32, 80, 16)
```

### A2

```text
ConvBlock(32 -> 32, residual=True)
```

Output:

```text
(B*T, 32, 80, 16)
```

### A3

```text
ConvBlock(32 -> 64, kernel=3, stride=(3, 1), padding=1)
```

Output:

```text
(B*T, 64, 27, 16)
```

### A4

```text
ConvBlock(64 -> 64, residual=True)
```

Output:

```text
(B*T, 64, 27, 16)
```

### A5

```text
ConvBlock(64 -> 64, residual=True)
```

Output:

```text
(B*T, 64, 27, 16)
```

### A6

```text
ConvBlock(64 -> 128, kernel=3, stride=3, padding=1)
```

Output:

```text
(B*T, 128, 9, 6)
```

### A7

```text
ConvBlock(128 -> 128, residual=True)
```

Output:

```text
(B*T, 128, 9, 6)
```

### A8

```text
ConvBlock(128 -> 128, residual=True)
```

Output:

```text
(B*T, 128, 9, 6)
```

### A9

```text
ConvBlock(128 -> 256, kernel=3, stride=(3, 2), padding=1)
```

Output:

```text
(B*T, 256, 3, 3)
```

### A10

```text
ConvBlock(256 -> 256, residual=True)
```

Output:

```text
(B*T, 256, 3, 3)
```

### A11

```text
ConvBlock(256 -> 512, kernel=3, stride=1, padding=0)
```

Output:

```text
(B*T, 512, 1, 1)
```

### A12

```text
ConvBlock(512 -> 512, kernel=1, stride=1, padding=0)
```

Output:

```text
(B*T, 512, 1, 1)
```

### Audio Tower Result

Final audio embedding:

```text
(B*T, 512, 1, 1)
```

This is the tensor that seeds the decoder.

## Face Decoder

The decoder starts from the audio embedding, then upsamples back to `96x96`.
After each decoder block it concatenates the corresponding face encoder skip.

### D0

Input:

```text
(B*T, 512, 1, 1)
```

Block:

```text
ConvBlock(512 -> 512, kernel=1, stride=1, padding=0)
```

Output before skip concat:

```text
(B*T, 512, 1, 1)
```

Concat with skip `E6`:

```text
512 + 512 = 1024 channels
```

Result:

```text
(B*T, 1024, 1, 1)
```

### D1

Input:

```text
(B*T, 1024, 1, 1)
```

Block:

```text
DeconvBlock(1024 -> 512, kernel=3, stride=1, padding=0, output_padding=0)
ConvBlock(512 -> 512, residual=True)
```

Spatial change:

```text
1x1 -> 3x3
```

Output before skip concat:

```text
(B*T, 512, 3, 3)
```

Concat with skip `E5`:

```text
512 + 512 = 1024
```

Result:

```text
(B*T, 1024, 3, 3)
```

### D2

Input:

```text
(B*T, 1024, 3, 3)
```

Block:

```text
DeconvBlock(1024 -> 512, kernel=3, stride=2, padding=1, output_padding=1)
ConvBlock(512 -> 512, residual=True)
ConvBlock(512 -> 512, residual=True)
```

Spatial change:

```text
3x3 -> 6x6
```

Output before skip concat:

```text
(B*T, 512, 6, 6)
```

Concat with skip `E4`:

```text
512 + 256 = 768
```

Result:

```text
(B*T, 768, 6, 6)
```

### D3

Input:

```text
(B*T, 768, 6, 6)
```

Block:

```text
DeconvBlock(768 -> 384, kernel=3, stride=2, padding=1, output_padding=1)
ConvBlock(384 -> 384, residual=True)
ConvBlock(384 -> 384, residual=True)
```

Spatial change:

```text
6x6 -> 12x12
```

Output before skip concat:

```text
(B*T, 384, 12, 12)
```

Concat with skip `E3`:

```text
384 + 128 = 512
```

Result:

```text
(B*T, 512, 12, 12)
```

### D4

Input:

```text
(B*T, 512, 12, 12)
```

Block:

```text
DeconvBlock(512 -> 256, kernel=3, stride=2, padding=1, output_padding=1)
ConvBlock(256 -> 256, residual=True)
ConvBlock(256 -> 256, residual=True)
```

Spatial change:

```text
12x12 -> 24x24
```

Output before skip concat:

```text
(B*T, 256, 24, 24)
```

Concat with skip `E2`:

```text
256 + 64 = 320
```

Result:

```text
(B*T, 320, 24, 24)
```

### D5

Input:

```text
(B*T, 320, 24, 24)
```

Block:

```text
DeconvBlock(320 -> 128, kernel=3, stride=2, padding=1, output_padding=1)
ConvBlock(128 -> 128, residual=True)
ConvBlock(128 -> 128, residual=True)
```

Spatial change:

```text
24x24 -> 48x48
```

Output before skip concat:

```text
(B*T, 128, 48, 48)
```

Concat with skip `E1`:

```text
128 + 32 = 160
```

Result:

```text
(B*T, 160, 48, 48)
```

### D6

Input:

```text
(B*T, 160, 48, 48)
```

Block:

```text
DeconvBlock(160 -> 64, kernel=3, stride=2, padding=1, output_padding=1)
ConvBlock(64 -> 64, residual=True)
ConvBlock(64 -> 64, residual=True)
```

Spatial change:

```text
48x48 -> 96x96
```

Output before skip concat:

```text
(B*T, 64, 96, 96)
```

Concat with skip `E0`:

```text
64 + 16 = 80
```

Result:

```text
(B*T, 80, 96, 96)
```

## Output Head

Current run uses `predict_alpha = false`, so only the RGB head is active.

### Face Output Head

```text
ConvBlock(80 -> 32, kernel=3, stride=1, padding=1)
Conv2d(32 -> 3, kernel=1, stride=1, padding=0)
Sigmoid()
```

Output:

```text
(B*T, 3, 96, 96)
```

### Alpha Head

Implemented, but not used in the current run:

```text
ConvBlock(80 -> 16, kernel=3, stride=1, padding=1)
Conv2d(16 -> 1, kernel=1, stride=1, padding=0)
Sigmoid()
```

## Output Reshaping Back To Sequence Form

If time was flattened on input, the output is restored as:

```text
(B*T, 3, 96, 96) -> split by B -> stack on time axis -> (B, 3, T, 96, 96)
```

So final generator output is:

```text
(B, 3, 5, 96, 96)
```

## Compact End-To-End Shape Trace

### Face Path

```text
6x96x96
-> 16x96x96
-> 32x48x48
-> 64x24x24
-> 128x12x12
-> 256x6x6
-> 512x3x3
-> 512x1x1
```

### Audio Path

```text
1x80x16
-> 32x80x16
-> 64x27x16
-> 128x9x6
-> 256x3x3
-> 512x1x1
```

### Decoder Path

```text
512x1
-> 512x1   + skip 512x1  => 1024x1
-> 512x3   + skip 512x3  => 1024x3
-> 512x6   + skip 256x6  => 768x6
-> 384x12  + skip 128x12 => 512x12
-> 256x24  + skip 64x24  => 320x24
-> 128x48  + skip 32x48  => 160x48
-> 64x96   + skip 16x96  => 80x96
-> 32x96
-> 3x96
```

## Relation To Official Wav2Lip

For `img_size == 96`, this path is architecturally very close to official `Wav2Lip`.

It matches the official model in:

- face encoder stage count
- spatial downsampling path
- audio tower structure
- decoder stage count
- skip concat pattern
- output head width (`80 -> 32 -> 3`)

So when we say "current almost-official `96x96` generator", this is what we mean:

- the generator topology is basically the official `96x96` topology
- most differences live outside the bare architecture:
  - training loop
  - loss setup
  - teacher usage
  - dataset pipeline

## Open Questions To Extend Later

Some natural next additions for this note:

- exact parameter count for the current `96x96` path
- per-layer receptive field intuition
- exact memory-heavy tensors during training
- where `SyncNet`, `L1`, and optional `GAN` attach to the generator output
- how this `96` path differs from the generalized `192` path

## Practical Note: First-Layer Options If We Move To `192x192`

This section is not part of the current `96x96` topology.
It is a practical design note for the specific question:

- if the face crop becomes `192x192`
- and the face occupies roughly the same fraction of the frame
- what should happen to the first visual stem layer?

We discussed three candidates:

1. keep `7x7`
2. replace the first layer with `13x13`
3. keep `7x7` and add another shallow layer near the input

### Option A: Keep `7x7`

Example:

```text
ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)
```

Pros:

- closest to the existing official-style design
- cheapest option
- easiest to reason about
- least likely to destabilize optimization

Cons:

- relative to the face size, the first receptive field becomes smaller than it was at `96x96`
- the first layer sees a more local piece of the mouth/face than before

When this is a good choice:

- when we want the safest baseline
- when we want to isolate the effect of higher resolution itself
- when we do not want to introduce a strong new architectural variable

### Option B: Replace The First Layer With `13x13`

Example:

```text
ConvBlock(6 -> 16, kernel=13, stride=1, padding=6)
```

Why this idea exists:

- if the same face becomes about `2x` larger in pixels, then `7x7` covers a smaller fraction of the face
- `13x13` is a reasonable odd-sized approximation to "double the receptive span"

Pros:

- better preserves the relative spatial coverage of the first layer
- gives the model broad local face context immediately
- conceptually matches the "same face, just denser pixels" intuition

Cons:

- much less standard than `7x7`
- first layer FLOPs rise strongly
- first-layer parameters rise by about `3.45x` versus `7x7`
- if results improve or worsen, it becomes harder to separate "resolution effect" from "big-kernel stem effect"

When this is a good choice:

- as a targeted ablation
- when we explicitly want scale-preserving first-layer behavior
- when we are willing to pay extra compute in the largest spatial layer

### Option C: Keep `7x7` And Add Another Shallow Layer

Example direction:

```text
ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)
ConvBlock(16 -> 16, kernel=3, stride=1, padding=1)
```

or

```text
ConvBlock(6 -> 16, kernel=7, stride=1, padding=3)
ConvBlock(16 -> 16, kernel=5, stride=1, padding=2)
```

Pros:

- keeps the familiar `7x7` stem
- grows receptive field more gradually
- more standard than jumping straight to `13x13`
- gives the network extra low-level processing capacity before the first downsample

Cons:

- this is no longer a clean "same architecture, larger image" comparison
- adds compute and activations at the largest resolution
- still does not preserve relative receptive-field size as directly as a single `13x13`

When this is a good choice:

- when we want a conservative way to enrich the high-resolution stem
- when we suspect `7x7` alone is too local, but `13x13` feels too aggressive

### Practical Recommendation

If the goal is:

- easiest and safest first experiment for `192`

then the order I would test is:

1. keep `7x7`
2. keep `7x7` plus one extra shallow layer
3. `13x13` as a deliberate ablation

Why this order:

- `7x7` gives the cleanest baseline
- `7x7 + extra shallow layer` is a softer architectural change than a big-kernel jump
- `13x13` is the most directly scale-preserving, but also the most unusual and the hardest to interpret cleanly

If the goal is instead:

- preserve first-layer face coverage as literally as possible

then `13x13, stride=1, padding=6` is the most direct match to that intuition.
