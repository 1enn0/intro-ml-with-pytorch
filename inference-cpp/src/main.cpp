//------------------------------------------------------------------------
// Project     : esc-app
// Filename    : main.cpp
// Created by  : lhannink, 01/2021
// Description : loading a pytorch model and running inference
//------------------------------------------------------------------------

#include <chrono>
#include <iostream>

#include <torch/script.h>
#include <torch/serialize.h>

#include "vocab-gen.h"

//------------------------------------------------------------------------
void print_usage ()
{
	std::cerr
		<< "Usage:\n"
		<< "\tesc-app <module-path> <inputs-path> [<outputs-path>]\n\n"
		<< "Arguments:\n"
		<< "\t module-path : path to TorchScript module to run\n"
		<< "\t inputs-path : path to TorchScript module containing one or more named buffers\n"
		<< "\t				 identical shape. These buffers will be stacked into a single tensor\n"
		<< "\t               that will be passed to the model's forward() method.\n"
		<< "\toutputs-path : (optional) where to save the model output. E.g., this can be \n"
		<< "\t               loaded again in Python by running\n"
		<< "\t               >>> import torch\n"
		<< "\t               >>> output_module = torch.load('path/to/outputs.pt')\n"
		<< "\t               >>> outputs = next(output_module.parameters())\n\n";
}

//------------------------------------------------------------------------
bool load_module (const std::string& filename, torch::jit::script::Module& module)
{
	std::cout << "Loading TorchScript module from '" << filename << "'...";
	try
	{
		module = torch::jit::load (filename);
		std::cout << "done.\n";
		return true;
	}
	catch (const c10::Error& e)
	{
		std::cerr << "error\n";
		return false;
	}
}

//------------------------------------------------------------------------
std::string get_shape (const at::Tensor& t)
{
	std::string shape{ "[" };
	for (int d{ 0 }; d < t.ndimension () - 1; ++d)
	{
		shape.append (std::to_string (t.size (d)));
		shape.append (", ");
	}
	shape.append (std::to_string (t.size (t.ndimension () - 1)));
	shape.append ("]");
	return shape;
}

//------------------------------------------------------------------------
int with_error_msg(const std::string& msg)
{
	std::cerr << msg << "\n";
	return -1;
}

//------------------------------------------------------------------------
int main (int argc, const char* argv[])
{
	std::cout << "==========================================\n";
	std::cout << "===   Environmental Sound Classifier   ===\n";
	std::cout << "==========================================\n\n";
	if (argc < 2)
	{
		print_usage();
		return -1;
	}

	// deserialize the script module from file
	torch::jit::script::Module model;
	if (!load_module(argv[1], model))
		return with_error_msg("failed");

	if (argc < 3)
		return 0;

	torch::jit::script::Module input_container;
	if (!load_module(argv[2], input_container))
		return with_error_msg("loading of module failed");

	std::cout << "\nInspecting '" << argv[2] << "' for named buffers.\n";
	auto n_samples = static_cast<int> (input_container.named_buffers().size ());
	auto inputs = torch::zeros ({n_samples, 1, 128, 157});
	std::vector<std::string> labels;

	int idx = 0;
	for (const auto& attr : input_container.named_buffers())
	{
		auto& tensor = attr.value;
		labels.push_back (attr.name);

		std::cout << "  - found buffer '" << attr.name << "' with shape " << get_shape(tensor) << "\n";

		// copy buffer data into stacked input tensor
		inputs.slice (0, idx, idx + 1) = tensor;
		idx++;
	}

	// run single inference step
	std::cout << "\nRunning inference using stacked input tensor with shape " << get_shape(inputs) << "...";

	const auto start = std::chrono::high_resolution_clock::now ();
	const auto outputs = model.forward ({inputs}).toTensor ();
	const auto duration = std::chrono::duration<double, std::milli> (std::chrono::high_resolution_clock::now () - start);
	std::cout << "took " << duration.count () << " ms.\n";

	// print model predictions vs labels
	auto max_idcs = outputs.argmax (1);
	std::cout << "\nOutput: [sample #] (prediction, label)\n";
	for (int i {0}; i < n_samples; ++i)
		std::cout << "  [#" << i + 1 << "] ('" << vocab[max_idcs[i].item<int> ()] << "', '" << labels[i]
		          << "')\n";

	// save output tensor for further processing back in python
	if (argc >= 4)
	{
		std::cout << "\nSaving output tensor to '" << argv[3] << "'.";
		torch::save (outputs, argv[3]);
	}

	return 0;
}
