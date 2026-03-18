#!/usr/bin/env python3
"""
Logical flow graph of one ray through the GPU path tracer.
Creates a clean, non-overlapping diagram. Tries Graphviz first, then matplotlib.
"""

import os

def create_graphviz_diagram():
    """Create diagram using Graphviz - automatic layout, no overlaps."""
    try:
        from graphviz import Digraph
    except ImportError:
        return False

    dot = Digraph(comment='Path Tracer Ray Flow', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.7', ranksep='0.9',
             fontname='Arial', fontsize='11', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', margin='0.2,0.15')
    dot.attr('edge', fontname='Arial', fontsize='9')

    # Process nodes (blue)
    dot.node('start', 'START\nPixel (i,j), RNG seed', fillcolor='#e3f2fd', color='#1565c0')
    dot.node('ray', 'Generate primary ray\norigin=camera, dir=invViewProj*(uv+jitter)', fillcolor='#e3f2fd', color='#1565c0')
    dot.node('trace', 'traceRayEXT(primary)\nsbtOffset=0, missIndex=0', fillcolor='#ede7f6', color='#5e35b1')
    dot.node('hit_check', 'hitT < 0?', shape='diamond', fillcolor='#fff3e0', color='#e65100')

    # Miss branch
    dot.node('env', 'Sample env map\ncolor += throughput * envColor', fillcolor='#e8f5e9', color='#2e7d32')
    dot.node('sky_gbuf', 'Write sky G-buffer\nviewZ=600000', fillcolor='#e8f5e9', color='#2e7d32')
    dot.node('exit_miss', 'Firefly clamp\nWrite outputs → EXIT', fillcolor='#ffebee', color='#c62828')

    # Hit branch
    dot.node('payload', 'Closest-hit wrote payload\nhitPos, N, albedo, roughness, emissive', fillcolor='#ede7f6', color='#5e35b1')
    dot.node('emissive', 'color += throughput * emissive\nFirst hit? → Write G-buffer', fillcolor='#e8f5e9', color='#2e7d32')
    dot.node('direct', 'Direct lighting\nJitter sun → traceRayEXT(shadow)\nmiss → color += BRDF * radiance * NdotL', fillcolor='#e8f5e9', color='#2e7d32')
    dot.node('bsdf', 'BSDF importance sampling\nFresnel → sample spec or diff\nthroughput *= bsdfWeight', fillcolor='#e8f5e9', color='#2e7d32')
    dot.node('bounce_check', 'bounce > 1?', shape='diamond', fillcolor='#fff3e0', color='#e65100')
    dot.node('rr_check', 'rand() > p?', shape='diamond', fillcolor='#fff3e0', color='#e65100')
    dot.node('next_bounce', 'origin = hitPos + N*ε\ndirection = newDir\n(throughput /= p if RR)', fillcolor='#c8e6c9', color='#1b5e20')
    dot.node('exit_loop', 'EXIT loop\n(RR break or max bounces)', fillcolor='#e3f2fd', color='#1565c0')
    dot.node('final', 'Firefly clamp (cMax>50)\nWrite colorOutput, accumBuffer', fillcolor='#e3f2fd', color='#1565c0')

    # Edges
    dot.edge('start', 'ray')
    dot.edge('ray', 'trace')
    dot.edge('trace', 'hit_check')
    dot.edge('hit_check', 'env', 'YES\n(miss)')
    dot.edge('hit_check', 'payload', 'NO\n(hit)')
    dot.edge('env', 'sky_gbuf')
    dot.edge('sky_gbuf', 'exit_miss')
    dot.edge('payload', 'emissive')
    dot.edge('emissive', 'direct')
    dot.edge('direct', 'bsdf')
    dot.edge('bsdf', 'bounce_check')
    dot.edge('bounce_check', 'next_bounce', 'NO')
    dot.edge('bounce_check', 'rr_check', 'YES')
    dot.edge('rr_check', 'exit_loop', 'YES')
    dot.edge('rr_check', 'next_bounce', 'NO\n(throughput/=p)')
    dot.edge('next_bounce', 'trace', 'loop')
    dot.edge('exit_loop', 'final')

    out_path = 'ray_flow_graph'
    dot.render(out_path, format='png', cleanup=True)
    print(f'Saved {out_path}.png (Graphviz)')
    return True

def create_matplotlib_diagram():
    """Matplotlib fallback: two-column layout to avoid overlaps."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon

    fig, ax = plt.subplots(1, 1, figsize=(12, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 26)
    ax.axis('off')

    # Two columns: left = miss path, right = hit path + main flow
    left_x, right_x = 2, 6
    box_w, box_h = 2.8, 0.75
    dy = 1.5

    def box(ax, x, y, text, fc='#e3f2fd', ec='#1565c0'):
        p = FancyBboxPatch((x, y), box_w, box_h, boxstyle='round,pad=0.25',
                           facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(p)
        ax.text(x + box_w/2, y + box_h/2, text, ha='center', va='center', fontsize=8)

    def diamond(ax, x, y, text, s=0.4):
        pts = [(x, y+s), (x+s, y), (x, y-s), (x-s, y)]
        p = Polygon(pts, facecolor='#fff3e0', edgecolor='#e65100', linewidth=2)
        ax.add_patch(p)
        ax.text(x, y, text, ha='center', va='center', fontsize=8)

    def arr(ax, a, b, lbl=''):
        ax.annotate(lbl, xy=b, xytext=a, fontsize=7,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    # Title
    ax.text(5, 25, 'Path Tracer: Logical Flow of One Ray', fontsize=14, ha='center', fontweight='bold')
    ax.text(5, 24.4, '(pt_raygen.rgen bounce loop)', fontsize=9, ha='center', color='#666')

    y = 23
    # Center column for shared start
    box(ax, right_x - box_w/2, y, 'START: Pixel (i,j), RNG seed')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+box_h))
    box(ax, right_x - box_w/2, y, 'Generate primary ray\norigin=camera, dir=invViewProj*(uv+jitter)')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+box_h))
    box(ax, right_x - box_w/2, y, 'traceRayEXT(primary)', '#ede7f6', '#5e35b1')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+0.4))
    diamond(ax, right_x, y, 'hitT < 0?')
    y -= 0.6

    # LEFT: Miss branch (separate column)
    arr(ax, (right_x-0.4, y+1.0), (left_x+box_w/2, y+0.2), 'YES\n(miss)')
    box(ax, left_x, y-0.6, 'Sample env map\ncolor += throughput * envColor', '#e8f5e9', '#2e7d32')
    arr(ax, (left_x+box_w/2, y-0.6), (left_x+box_w/2, y-1.4))
    box(ax, left_x, y-2.0, 'Write sky G-buffer\nviewZ=600000', '#e8f5e9', '#2e7d32')
    arr(ax, (left_x+box_w/2, y-2.0), (left_x+box_w/2, y-2.8))
    box(ax, left_x, y-3.4, 'Firefly clamp\nWrite outputs → EXIT', '#ffebee', '#c62828')

    # RIGHT: Hit branch
    arr(ax, (right_x+0.4, y+1.0), (right_x, y-0.2), 'NO\n(hit)')
    y -= dy
    box(ax, right_x - box_w/2, y, 'Closest-hit wrote payload\nhitPos, N, albedo, roughness, emissive', '#ede7f6', '#5e35b1')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+box_h))
    box(ax, right_x - box_w/2, y, 'color += throughput * emissive\nFirst hit? → Write G-buffer', '#e8f5e9', '#2e7d32')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+box_h))
    box(ax, right_x - box_w/2, y, 'Direct lighting\nJitter sun → traceRayEXT(shadow)\nmiss → color += BRDF * radiance * NdotL', '#e8f5e9', '#2e7d32')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+box_h))
    box(ax, right_x - box_w/2, y, 'BSDF importance sampling\nFresnel → sample spec/diff\nthroughput *= bsdfWeight', '#e8f5e9', '#2e7d32')
    y -= dy
    arr(ax, (right_x, y+dy), (right_x, y+0.4))
    diamond(ax, right_x, y, 'bounce > 1?')
    y -= 0.6

    # Bounce NO → next bounce
    arr(ax, (right_x-0.4, y+0.6), (right_x-0.5, y-0.3), 'NO')
    arr(ax, (right_x-0.5, y-0.3), (right_x, y-0.3))
    box(ax, right_x - box_w/2, y-1.0, 'origin = hitPos + N*ε\ndirection = newDir', '#c8e6c9', '#1b5e20')
    arr(ax, (right_x, y-0.3), (right_x, y-0.7))
    # Loop back to trace
    y_trace = 15.2
    ax.annotate('', xy=(right_x, y_trace), xytext=(right_x, y-1.0),
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=2))
    ax.text(right_x+0.5, (y_trace + y-1.0)/2, 'loop', fontsize=8, color='#2e7d32')

    # Bounce YES → Russian roulette (right side)
    rr_x = 8.2
    arr(ax, (right_x+0.4, y+0.6), (rr_x-box_w/2+0.3, y+0.2), 'YES')
    diamond(ax, rr_x, y, 'rand > p?')
    arr(ax, (rr_x+0.4, y), (rr_x+0.2, y-1.0), 'YES')
    box(ax, rr_x - box_w/2, y-1.5, 'EXIT loop\n(RR or max bounces)', '#e3f2fd', '#1565c0')
    arr(ax, (rr_x, y-1.0), (rr_x, y-1.2))
    arr(ax, (rr_x+box_w/2-0.3, y-1.5), (right_x+0.5, y-2.5))
    box(ax, right_x - box_w/2, y-3.2, 'Firefly clamp (cMax>50)\nWrite colorOutput, accumBuffer', '#e3f2fd', '#1565c0')
    arr(ax, (right_x+0.5, y-2.5), (right_x, y-2.8))

    # RR NO → next bounce
    arr(ax, (rr_x-0.4, y), (right_x+0.5, y-0.3), 'NO')
    arr(ax, (right_x+0.5, y-0.3), (right_x, y-0.3))

    plt.tight_layout()
    plt.savefig('ray_flow_graph.png', dpi=200, bbox_inches='tight', facecolor='white')
    print('Saved ray_flow_graph.png (matplotlib)')
    plt.close()

def main():
    if create_graphviz_diagram():
        return
    print('Graphviz not available (install: pip install graphviz, and Graphviz binary from graphviz.org)')
    print('Using matplotlib fallback...')
    create_matplotlib_diagram()

if __name__ == '__main__':
    main()
