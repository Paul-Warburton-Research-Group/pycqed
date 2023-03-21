#
#
#
#
#
#

# List of recognized latex symbols
supported_symbols = {
        'alpha':                     '\\alpha',
        'beta':                      '\\beta',
        'gamma':                     '\\gamma',
        'delta':                     '\\delta',
        'epsilon':                   '\\epsilon',
        'varepsilon':                '\\varepsilon',
        'zeta':                      '\\zeta',
        'eta':                       '\\eta',
        'theta':                     '\\theta',
        'vartheta':                  '\\vartheta',
        'kappa':                     '\\kappa',
        'lambda':                    '\\lambda',
        'mu':                        '\\mu',
        'nu':                        '\\nu',
        'xi':                        '\\xi',
        'pi':                        '\\pi',
        'varpi':                     '\\varpi',
        'rho':                       '\\rho',
        'varrho':                    '\\varrho',
        'sigma':                     '\\sigma',
        'varsigma':                  '\\varsigma',
        'tau':                       '\\tau',
        'upsilon':                   '\\upsilon',
        'phi':                       '\\phi',
        'varphi':                    '\\varphi',
        'chi':                       '\\chi',
        'psi':                       '\\psi',
        'omega':                     '\\omega',
        
        'Gamma':                     '\\Gamma',
        'Delta':                     '\\Delta',
        'Theta':                     '\\Theta',
        'Lambda':                    '\\Lambda',
        'Xi':                        '\\Xi',
        'Pi':                        '\\Pi',
        'Sigma':                     '\\Sigma',
        'Upsilon':                   '\\Upsilon',
        'Phi':                       '\\Phi',
        'Psi':                       '\\Psi',
        'Omega':                     '\\Omega'
}

# Attempt to turn a string into a latex code
def latexify_param_name(param_string):
    
    latex_string = ""
    
    # Check if param_string starts with a latex symbol name
    found = False
    for s in list(supported_symbols.keys()):
        if param_string.rfind(s) == 0:
            found = True
            break
    
    # If found then separate it and latexify it
    if found:
        latex_string = supported_symbols[s] + "_\\mathrm{" + param_string[len(s):] + "}"
    else:
        latex_string = param_string[0] + "_\\mathrm{" + param_string[1:] + "}"
    
    return latex_string




