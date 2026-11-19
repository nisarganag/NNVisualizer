import * as nn from './nn';

export interface ModelState {
  nodes: Array<{ id: string; bias: number }>;
  links: Array<{ id: string; weight: number; isDead: boolean }>;
}

export function captureModel(network: nn.Node[][]): ModelState {
  const model: ModelState = { nodes: [], links: [] };
  nn.forEachNode(network, true, node => {
    model.nodes.push({ id: node.id, bias: node.bias });
    node.inputLinks.forEach(link => model.links.push({
      id: link.id, weight: link.weight, isDead: link.isDead,
    }));
  });
  return model;
}

export function applyModel(network: nn.Node[][], model: ModelState): void {
  const nodes = new Map(model.nodes.map(node => [node.id, node]));
  const links = new Map(model.links.map(link => [link.id, link]));
  nn.forEachNode(network, true, node => {
    const values = nodes.get(node.id);
    if (!values) throw new Error(`Missing neuron ${node.id}`);
    node.bias = values.bias;
    node.inputLinks.forEach(link => {
      const values = links.get(link.id);
      if (!values) throw new Error(`Missing connection ${link.id}`);
      link.weight = values.weight;
      link.isDead = values.isDead;
    });
  });
}
