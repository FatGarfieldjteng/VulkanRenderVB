#!/usr/bin/env python3
"""
Logical flow graph of one ray through the GPU path tracer.
Uses matplotlib for drawing.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Figure setup
fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 22)
ax.set_aspect('equal')
ax.axis('off')

# Style
box_style = dict(boxstyle='round,pad=0.4', facecolor='#e8f4f8', edgecolor='#2c5aa0', linewidth=2)
diamond_style = dict(boxstyle='round,pad=0.3', facecolor='#fff4e6', edgecolor='#c45a11', linewidth=2)
loop_style = dict(boxstyle='round,pad=0.4', facecolor='#e8f8e8', edgecolor='#2d7a2d', linewidth=2)
shader_style = dict(boxstyle='round,pad=0.3', facecolor='#f0e6ff', edgecolor='#5a2dc4', linewidth=2)

def add_box(ax, x, y, w, h, text, style=box_style):
    p = FancyBboxPatch((x, y), w, h, **style)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, wrap=True)

def add_diamond(ax, x, y, size, text):
    from matplotlib.patches import Polygon
    pts = [(x, y+size), (x+size, y), (x, y-size), (x-size, y)]
    p = Polygon(pts, facecolor='#fff4e6', edgecolor='#c45a11', linewidth=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha='center', va='center', fontsize=8)

def add_arrow(ax, start, end, color='#333', style='->'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=2))

# Title
ax.text(5, 21, 'Path Tracer: Logical Flow of One Ray', fontsize=16, ha='center', fontweight='bold')
ax.text(5, 20.3, '(pt_raygen.rgen bounce loop)', fontsize=10, ha='center', color='#555')

# Flow nodes - top to bottom
y = 19
dy = 1.4

# 1. Start
add_box(ax, 3.5, y, 3, 0.8, 'START: Pixel (i,j)\nRNG seed from pixel + sampleOff', box_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+0.8))

# 2. Primary ray
add_box(ax, 3.2, y, 3.6, 0.9, 'Generate primary ray\norigin=camera, dir=invViewProj*(uv+jitter)', box_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+0.9))

# 3. traceRayEXT (primary)
add_box(ax, 3.0, y, 4.0, 0.9, 'traceRayEXT(primary)\nsbtOffset=0, missIndex=0', shader_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+0.9))

# 4. Hit or Miss?
add_diamond(ax, 5, y, 0.5, 'hitT\n< 0?')
y -= dy
add_arrow(ax, (5, y+dy+0.5), (5, y+0.5))

# 5a. Miss branch (left)
ax.text(2.2, y-0.3, 'YES\n(miss)', fontsize=8, ha='center', color='#c45a11')
add_arrow(ax, (4.5, y+0.5), (2.5, y-0.5))
add_box(ax, 1.0, y-1.5, 3.0, 0.9, 'Sample env map\ncolor += throughput * envColor', box_style)
add_arrow(ax, (2.5, y-0.5), (2.5, y-0.6))
add_box(ax, 1.0, y-2.8, 3.0, 0.8, 'Write sky G-buffer\nviewZ=600000', box_style)
add_arrow(ax, (2.5, y-1.5), (2.5, y-1.6))
add_box(ax, 1.0, y-4.0, 3.0, 0.8, 'Firefly clamp, write outputs\n→ EXIT', box_style)
add_arrow(ax, (2.5, y-2.8), (2.5, y-3.2))

# 5b. Hit branch - continue down
ax.text(6.5, y-0.3, 'NO\n(hit)', fontsize=8, ha='center', color='#2d7a2d')
add_arrow(ax, (5.5, y+0.5), (6.5, y-0.3))
add_arrow(ax, (6.5, y-0.3), (5, y-0.5))
y -= dy

# 6. Closest-hit wrote payload
add_box(ax, 3.0, y, 4.0, 0.8, 'Closest-hit wrote payload\nhitPos, N, albedo, roughness, metallic, emissive', shader_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+0.8))

# 7. Shading
add_box(ax, 3.2, y, 3.6, 0.9, 'color += throughput * emissive\nFirst hit? → Write G-buffer (normal, albedo, depth, motion)', loop_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+0.9))

# 8. Direct lighting
add_box(ax, 2.8, y, 4.4, 1.0, 'Direct lighting: jitter sun dir L\nNdotL>0? → traceRayEXT(shadow)\n  miss → color += BRDF * sunRadiance * NdotL', loop_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+1.0))

# 9. BSDF sample
add_box(ax, 3.0, y, 4.0, 1.0, 'BSDF importance sampling\npSpec from Fresnel → sample spec or diff\nnewDir, bsdfWeight, throughput *= bsdfWeight', loop_style)
y -= dy
add_arrow(ax, (5, y+dy), (5, y+1.0))

# 10. Russian roulette (bounce > 1 only)
add_diamond(ax, 5, y, 0.45, 'bounce\n>1?')
y_rr = y
y -= dy
add_arrow(ax, (5, y+dy+0.45), (5, y_rr+0.45))
# NO: go to Next bounce
ax.text(3.2, y_rr-0.1, 'NO', fontsize=7, ha='right', color='#2d7a2d')
add_arrow(ax, (4.55, y_rr), (3.8, y_rr-0.3))
# YES: Russian roulette
ax.text(6.5, y_rr+0.1, 'YES', fontsize=7, ha='left', color='#c45a11')
add_arrow(ax, (5.45, y_rr), (6.2, y_rr))
add_diamond(ax, 6.8, y_rr-0.5, 0.35, 'rand\n>p?')
add_arrow(ax, (6.2, y_rr), (6.45, y_rr-0.5))
# rand>p → EXIT; rand<=p → throughput/=p, Next bounce
y_next = y - 0.9
add_box(ax, 3.2, y_next, 3.6, 0.8, 'origin = hitPos + N*ε\ndirection = newDir\n(throughput/=p if RR)', loop_style)
add_arrow(ax, (5, y_rr+0.45), (5, y_rr+0.05))
add_arrow(ax, (3.8, y_rr-0.3), (3.8, y_next+0.6))
add_arrow(ax, (3.8, y_next+0.6), (5, y_next+0.4))
# RR rand<=p: go to Next bounce
add_arrow(ax, (6.45, y_rr-0.5), (6.2, y_next+0.4))
add_arrow(ax, (6.2, y_next+0.4), (5, y_next+0.4))

# Loop back to traceRayEXT (y ≈ 15.2)
y_trace = 15.25
ax.plot([5, 5.8], [y_next, y_next], color='#2d7a2d', lw=2)
ax.annotate('', xy=(5.8, y_trace), xytext=(5.8, y_next),
            arrowprops=dict(arrowstyle='->', color='#2d7a2d', lw=2))
ax.annotate('', xy=(5, y_trace), xytext=(5.8, y_trace),
            arrowprops=dict(arrowstyle='->', color='#2d7a2d', lw=2))
ax.plot([5, 5.8], [y_trace, y_trace], color='#2d7a2d', lw=2)
ax.annotate('', xy=(5, y_next), xytext=(5, y_trace),
            arrowprops=dict(arrowstyle='->', color='#2d7a2d', lw=2))
ax.text(6.0, (y_next + y_trace) / 2 - 0.3, 'loop', fontsize=8, color='#2d7a2d')

# 12. EXIT loop (RR break or max bounces)
add_box(ax, 6.0, y_next - 1.4, 2.8, 0.8, 'EXIT loop\n(RR break or max bounces)', box_style)
# rand>p from RR diamond → EXIT loop
add_arrow(ax, (7.15, y_rr-0.5), (7.2, y_next - 1.0))
add_arrow(ax, (7.2, y_next - 1.0), (6.9, y_next - 1.0))
# From EXIT to final output
add_box(ax, 3.5, y_next - 2.8, 3.0, 0.8, 'Firefly clamp (cMax>50)\nWrite colorOutput, accumBuffer', box_style)
add_arrow(ax, (6.5, y_next - 1.4), (5, y_next - 2.2))
add_arrow(ax, (5, y_next - 2.2), (5, y_next - 2.4))

# Legend
ax.text(0.5, 1.5, 'Shaders: pt_raygen.rgen (main), pt_closesthit.rchit, pt_miss.rmiss, pt_shadow_miss.rmiss',
        fontsize=8, style='italic', color='#666')
ax.text(0.5, 0.8, 'traceRayEXT is synchronous: payload available when it returns.',
        fontsize=8, style='italic', color='#666')

plt.tight_layout()
out_path = 'ray_flow_graph.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved {out_path}')
# plt.show()  # Uncomment to display interactively
plt.close()
