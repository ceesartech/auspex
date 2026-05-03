import '@testing-library/jest-dom';
import { render, screen, fireEvent } from '@testing-library/react';
import { Input } from '@/components/ui/input';

describe('Input', () => {
  it('renders input element', () => {
    render(<Input placeholder="Enter text" />);
    expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
  });

  it('shows error message', () => {
    render(<Input error="This field is required" />);
    expect(screen.getByText('This field is required')).toBeInTheDocument();
  });

  it('applies error styling', () => {
    render(<Input error="Error" data-testid="input" />);
    expect(screen.getByTestId('input')).toHaveClass('border-destructive');
  });

  it('handles value changes', () => {
    const onChange = jest.fn();
    render(<Input onChange={onChange} placeholder="Type here" />);
    fireEvent.change(screen.getByPlaceholderText('Type here'), {
      target: { value: 'test' },
    });
    expect(onChange).toHaveBeenCalled();
  });

  it('can be disabled', () => {
    render(<Input disabled placeholder="Disabled" />);
    expect(screen.getByPlaceholderText('Disabled')).toBeDisabled();
  });
});
